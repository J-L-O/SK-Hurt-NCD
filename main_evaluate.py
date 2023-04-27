import argparse
import fnmatch
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, List, Union, Dict

from utils.data import get_datamodule

from utils.metrics import compute_metrics
from utils.mmd import mmd_cal, compute_transfer_flow, bootstrap_transfer_flow
from utils.nets import MultiHeadResNet


if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    port = int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT', "12034"))
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True, suspend=False)

parser = ArgumentParser()
parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
parser.add_argument('--data_dir', type=str, default='~/code/Data/')
parser.add_argument("--download", default=False, action="store_true", help="download dataset if not exists")
parser.add_argument("--imagenet_subset", default="all", type=str, help="imagenet subset ('all' or BREEDS dataset)")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument('--exp_dir', type=str, default='experiments/')
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=0, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--intermediate_keys", nargs="*", help="intermediate layers to extract")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=1, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--model_path", type=str, help="path to the trained model. May contain a wildcard for the epoch")
parser.add_argument("--from_pretrained", default=False, action="store_true", help="from supervised pretrained model")
parser.add_argument("--from_swav", default=False, action="store_true", help="from SwAV pretrained model")
parser.add_argument('--transfer_flow', default=False, action="store_true", help="compute transfer flow")
parser.add_argument('--kernel', default="gaussian", type=str, help="kernel function to use for transfer flow")
parser.add_argument('--return_bandwidths', default=False, action="store_true", help="return mmd values for each bandwidth")
parser.add_argument('--bootstrap_transfer_flow', default=False, action="store_true", help="bootstrap transfer flow")
parser.add_argument("--bootstrap_num_samples", default=10, type=int, help="bootstrap sample number")
parser.add_argument('--pseudo_transfer_flow', default=False, action="store_true",
                    help="compute pseudo transfer flow")
parser.add_argument('--clustering_metrics', default=False, action="store_true", help="compute ACC, NMI and ARI")
parser.add_argument('--disable_tqdm', default=False, action="store_true", help="disable tqdm progress bar")


def load_model(args: argparse.Namespace, checkpoint: Path, device: torch.device,
               hook_keys: List[str] = None) -> nn.Module:
    model = MultiHeadResNet(
        arch=args.arch,
        low_res="CIFAR" in args.dataset,
        num_labeled=args.num_labeled_classes,
        num_unlabeled=args.num_unlabeled_classes,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        overcluster_factor=args.overcluster_factor,
        num_heads=args.num_heads,
        num_hidden_layers=args.num_hidden_layers,
        hook_keys=hook_keys
    ).to(device)

    if args.from_pretrained:
        state_dict = torch.load(checkpoint, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        model.load_state_dict(state_dict, strict=False)
    elif args.from_swav:
        state_dict = torch.load(checkpoint, map_location=device)
        state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        assert all(["head" in key for key in missing]), f"Missing keys {missing}"
        assert all([("head" in key or "prototypes" in key) for key in unexpected]), f"Unexpected keys {unexpected}"
    else:
        state_dict = torch.load(checkpoint, map_location=device)["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if ("model" in k)}
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()

    return model


@torch.no_grad()
def compute_features(model: nn.Module, device: torch.device, loader: DataLoader,
                     feature_keys: List[Union[str, int]], disable_tqdm: bool) -> Tuple[Dict, np.ndarray]:
    features = {}
    labels = np.zeros(len(loader.dataset), dtype=np.int64)

    start = 0

    for batch_idx, (x, label) in enumerate(tqdm(loader, disable=disable_tqdm)):
        x, label = x.to(device), label.to(device)
        outputs = model(x)

        end = start + loader.batch_size

        for key in feature_keys:
            feature_shape = outputs[key].shape

            # In case we are dealing with multiple heads we just take the output of the first
            if len(feature_shape) == 3:
                outputs[key] = outputs[key][0]
                feature_shape = outputs[key].shape

            if key not in features.keys():
                features[key] = np.zeros((len(loader.dataset), feature_shape[-1]))

            features[key][start:end, :] = outputs[key].cpu().numpy()
            labels[start:end] = label.cpu().numpy()

        start = end

    return features, labels


@torch.no_grad()
def compute_labeled_vs_unlabeled(features_labeled: np.ndarray, labels_labeled: np.ndarray,
                                 features_unlabeled: np.ndarray, labels_unlabeled: np.ndarray,
                                 num_labeled_classes: int, num_unlabeled_classes: int,
                                 device: torch.device) -> np.ndarray:
    mmd_matrix = torch.zeros((num_labeled_classes, num_unlabeled_classes), dtype=torch.float32, device=device)

    for i in range(num_labeled_classes):
        for j in range(num_unlabeled_classes):
            indices_i = labels_labeled == i
            indices_j = labels_unlabeled == j

            features_i = torch.tensor(features_labeled[indices_i], device=device)
            features_j = torch.tensor(features_unlabeled[indices_j], device=device)

            mmd = mmd_cal(features_i, features_j)

            mmd_matrix[i, j] = mmd

    return mmd_matrix.cpu().numpy()


def evaluate_model(args: argparse.Namespace, checkpoint: Path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using {device}")

    dm = get_datamodule(args, "evaluate")
    dm.setup()
    labeled_loader, unlabeled_loader = dm.train_dataloader()

    if args.intermediate_keys is None:
        args.intermediate_keys = []
    unlabeled_keys = ["proj_feats_unlab", "logits_lab", "logits_unlab", "feats"] + args.intermediate_keys
    labeled_keys = ["feats", "logits_lab"] + args.intermediate_keys

    print(f"Loading {args.arch}")
    model = load_model(args, checkpoint, device, hook_keys=args.intermediate_keys)

    print(f"Computing model outputs")
    features_unlabeled, labels_unlabeled = compute_features(model, device, unlabeled_loader, unlabeled_keys, args.disable_tqdm)
    labels_unlabeled = labels_unlabeled - args.num_labeled_classes  # For convenient handling

    features_labeled, labels_labeled = compute_features(model, device, labeled_loader, labeled_keys, args.disable_tqdm)

    del model
    torch.cuda.empty_cache()

    if args.clustering_metrics:
        print(f"Computing clustering metrics")
        preds_unlabeled_set = features_unlabeled["logits_unlab"].argmax(axis=1)
        acc, nmi, ari = compute_metrics(preds_unlabeled_set, labels_unlabeled)
        print(f"Clustering metrics are: {acc}, {nmi}, {ari}")

    if args.bootstrap_transfer_flow:
        print(f"Computing bootstrapped transfer flow")
        num_samples = args.bootstrap_num_samples

        mean, std, standard_error = bootstrap_transfer_flow(features_unlabeled["feats"], labels_unlabeled, num_samples,
                                                            args.kernel, args.num_unlabeled_classes, device,
                                                            return_all=args.return_bandwidths)

        print(f"Unlabeled bootstrapped transfer flow is: {mean} +- {standard_error}")

    if args.transfer_flow:
        print(f"Computing  transfer flow")
        unlabeled_transfer_flow = compute_transfer_flow(features_unlabeled["feats"], labels_unlabeled,
                                                           args.kernel, args.num_unlabeled_classes, device,
                                                           return_all=args.return_bandwidths)

        print(f"Unlabeled transfer flow is: {unlabeled_transfer_flow}")

    if args.pseudo_transfer_flow:
        clustering_methods = {
            "kmeans": KMeans(n_clusters=args.num_unlabeled_classes, init='k-means++'),
            "gmm": GaussianMixture(n_components=args.num_unlabeled_classes),
            "agglomerative": AgglomerativeClustering(n_clusters=args.num_unlabeled_classes),
        }

        for method_name, clustering_method in clustering_methods.items():
            print(f"Computing pseudo transfer flow with {method_name}")
            pseudo_labels = clustering_method.fit_predict(features_unlabeled["feats"])
            pseudo_transfer_flow = compute_transfer_flow(features_unlabeled["feats"], pseudo_labels,
                                                            args.kernel, args.num_unlabeled_classes, device,
                                                            return_all=args.return_bandwidths)

            print(f"Pseudo transfer flow with {method_name} is: {pseudo_transfer_flow}")


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    model_path = Path(args.model_path)

    print(f"Evaluating checkpoint {model_path.name}")
    evaluate_model(args, model_path)

