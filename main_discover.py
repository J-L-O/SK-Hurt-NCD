import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics, ClusterAccuracy
from utils.sinkhorn_knopp import SinkhornKnopp

import numpy as np
from argparse import ArgumentParser
from datetime import datetime

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    port = int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT', "12034"))
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True, suspend=False)

parser = ArgumentParser()
parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
parser.add_argument("--imagenet_subset", default="all", type=str, help="imagenet subset ('all' or BREEDS dataset)")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--download", default=False, action="store_true", help="whether to download")
parser.add_argument("--data_dir", default="/path/to/dataset", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
parser.add_argument("--checkpoint_freq", default=10, type=int, help="checkpoint frequency")
parser.add_argument("--batch_size", default=512, type=int, help="batch size")
parser.add_argument("--num_workers", default=0, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, default="/path/to/pretrained/model.ckpt")
parser.add_argument("--from_swav", default=False, action="store_true", help="start from SwAV model")
parser.add_argument("--unsupervised", default=False, action="store_true", help="don't use labels at all")
parser.add_argument("--unlabeled_data_only", default=False, action="store_true", help="only use unlabeled data")
parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
parser.add_argument("--mix_sup_selfsup", default=False, action="store_true", help="mix supervised and selfsupervised")
parser.add_argument("--mix_sup_weight", default=0.5, type=float, help="supervised weight")


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        num_labeled_classes = self.hparams.num_labeled_classes

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset,
            num_labeled=num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers
        )

        if self.hparams.from_swav:
            state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
            state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            assert all(["head" in key for key in missing]), f"Missing keys {missing}"
            assert all([("head" in key or "prototypes" in key) for key in unexpected]), f"Unexpected keys {unexpected}"
        else:
            state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
            state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}

            self.model.load_state_dict(state_dict, strict=False)

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )

        # metrics
        metrics_list = [ClusterMetrics(self.hparams.num_heads), ClusterMetrics(self.hparams.num_heads)]
        metrics_inc_list = [ClusterMetrics(self.hparams.num_heads), ClusterMetrics(self.hparams.num_heads)]

        if not self.hparams.unlabeled_data_only:
            metrics_list.append(ClusterAccuracy() if self.hparams.unsupervised else Accuracy())
            metrics_inc_list.append(ClusterAccuracy() if self.hparams.unsupervised else Accuracy())

        self.metrics = torch.nn.ModuleList(metrics_list)
        self.metrics_inc = torch.nn.ModuleList(metrics_inc_list)

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

        self.training_size = 500 * (self.hparams.num_labeled_classes + self.hparams.num_unlabeled_classes)
        self.steps = int(self.training_size / args.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )

        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_large_crops):
            for other_view in np.delete(range(self.hparams.num_crops), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_large_crops * (self.hparams.num_crops - 1))

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        if self.hparams.dataset == "ImageNet" and not self.hparams.unlabeled_data_only:
            views_lab, labels_lab, views_unlab, labels_unlab = batch
            views = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
            labels = torch.cat([labels_lab, labels_unlab])
            mask_lab = labels < self.hparams.num_labeled_classes
        else:
            views, labels = batch
            mask_lab = labels < self.hparams.num_labeled_classes

        return views, labels, mask_lab

    def training_step(self, batch, step):
        views, labels, mask_lab = self.unpack_batch(batch)

        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(views)
        results = {}

        if self.hparams.unlabeled_data_only:
            logits = outputs["logits_unlab"]
            logits_over = outputs["logits_unlab_over"]

            targets = torch.zeros_like(logits)
            targets_over = torch.zeros_like(logits_over)

            # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
            for v in range(self.hparams.num_crops):
                for h in range(self.hparams.num_heads):
                    targets_unlab = self.sk(outputs["logits_unlab"][v, h]).type_as(targets)
                    targets[v, h] = targets_unlab
                    targets_over[v, h] = self.sk(outputs["logits_unlab_over"][v, h]).type_as(targets)
        else:
            # gather outputs
            outputs["logits_lab"] = (
                outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)
            )
            outputs["logits_lab_over"] = outputs["logits_lab"]

            logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
            logits_over = torch.cat([outputs["logits_lab_over"], outputs["logits_unlab_over"]], dim=-1)

            # create targets
            targets_lab = (
                F.one_hot(labels[mask_lab], num_classes=nlc)
                .float()
                .to(self.device)
            )
            targets = torch.zeros_like(logits)
            targets_over = torch.zeros_like(logits_over)

            # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
            for v in range(self.hparams.num_crops):
                for h in range(self.hparams.num_heads):
                    if self.hparams.unsupervised:
                        targets_lab_sk = self.sk(outputs["logits_lab"][v, h, mask_lab]).type_as(targets)
                        targets_lab_over_sk = self.sk(outputs["logits_lab_over"][v, h, mask_lab]).type_as(targets)
                        targets[v, h, mask_lab, :nlc] = targets_lab_sk
                        targets_over[v, h, mask_lab, :nlc] = targets_lab_over_sk
                    elif self.hparams.mix_sup_selfsup:
                        targets_lab_sk = self.sk(outputs["logits_lab"][v, h, mask_lab]).type_as(targets)
                        targets_lab_over_sk = self.sk(outputs["logits_lab_over"][v, h, mask_lab]).type_as(targets)
                        targets_lab_gt = targets_lab.type_as(targets)

                        alpha = self.hparams.mix_sup_weight
                        targets[v, h, mask_lab, :nlc] = alpha * targets_lab_gt + (1 - alpha) * targets_lab_sk
                        targets_over[v, h, mask_lab, :nlc] = alpha * targets_lab_gt + (1 - alpha) * targets_lab_over_sk

                        kl = torch.nn.functional.kl_div(targets_lab_sk.log(), targets_lab_gt, reduction="batchmean")
                        results["kl_sup_selfsup"] = kl / (v * h)
                    else:
                        targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                        targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)

                    targets_unlab = self.sk(outputs["logits_unlab"][v, h, ~mask_lab]).type_as(targets)
                    targets[v, h, ~mask_lab, nlc:] = targets_unlab
                    targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                        outputs["logits_unlab_over"][v, h, ~mask_lab]
                    ).type_as(targets)

        # compute swapped prediction loss
        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # total loss
        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()

        loss = (loss_cluster + loss_overcluster) / 2

        results["loss"] = loss.detach()
        results["loss_cluster"] = loss_cluster.mean()
        results["loss_overcluster"] = loss_overcluster.mean()
        results["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dl_idx):
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        images, labels = batch

        # forward
        outputs = self(images)  # logitis_lab, logitis_unlab, proj_feats_unlab, logitis_unlab_over, proj_feats_unlab_over

        if "unlab" in tag:  # use clustering head
            preds = outputs["logits_unlab"]  # 256*20


            logits_lab = outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)

            preds_inc = torch.cat(           # 256*100
                [
                    logits_lab,
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            best_head = torch.argmin(self.loss_per_head)

            logits_lab = outputs["logits_lab"]

            preds = logits_lab
            preds_inc = torch.cat(
                [logits_lab, outputs["logits_unlab"][best_head]], dim=-1
            )
        preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)  ## all

    def validation_epoch_end(self, _):
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix

            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result)
                self.log(prefix_inc + "/acc", result_inc)


def main(args):
    dm = get_datamodule(args, "discover")

    run_name = "-".join(["discover", args.arch, args.dataset, args.comment])
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=run_name
    )

    model = Discoverer(**args.__dict__)
    checkpoint_filename = f'{args.arch}-{args.dataset}-{args.comment}-{{epoch}}'
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, save_top_k=-1,
                                          filename=checkpoint_filename, period=args.checkpoint_freq)

    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)
