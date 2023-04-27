import os

import pytorch_lightning as pl
from pl_bolts.callbacks import SSLOnlineEvaluator
from pl_bolts.models.self_supervised import SwAV

from utils.data import get_datamodule
from utils.callbacks import PretrainCheckpointCallback

from argparse import ArgumentParser
from datetime import datetime

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    port = int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT', "12034"))
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True, suspend=False)

parser = ArgumentParser()
parser.add_argument("--imagenet_subset", default="all", type=str, help="imagenet subset ('all' or 'entity30')")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--unlabeled_data_only", default=False, action="store_true", help="only use unlabeled data")
parser.add_argument("--labeled_data_only", default=False, action="store_true", help="only use labeled data")
parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")


def main(args):
    # build datamodule
    dm = get_datamodule(args, "pretrainSelfsupervised")

    # logger
    run_name = "-".join(["pretrainSwAV", args.arch, args.dataset, args.comment])
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=run_name
    )

    # Use low res ResNet for CIFAR
    if "CIFAR" in args.dataset:
        args.first_conv = False
        args.maxpool1 = False
    else:
        args.first_conv = True
        args.maxpool1 = True

    args.num_samples = dm.num_train_samples

    args.nmb_crops = [args.num_large_crops, args.num_small_crops]

    online_evaluator = SSLOnlineEvaluator(
        drop_p=0.0,
        hidden_dim=None,
        z_dim=args.hidden_mlp,
        num_classes=args.num_classes if not args.unlabeled_data_only else args.num_unlabeled_classes,
        dataset=args.dataset
    )

    checkpoint_callback = PretrainCheckpointCallback()

    callbacks = [online_evaluator, checkpoint_callback]

    model = SwAV(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=callbacks
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = SwAV.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)
