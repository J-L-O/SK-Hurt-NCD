import torch
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import CIFAR100

from utils.cifar import get_targets
from utils.datasets import ImageNetSubset
from utils.transforms import get_transforms

import numpy as np
import os


def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "ImageNet":
            return PretrainImageNetDataModule(args)
        elif "CIFAR100Split_" in args.dataset:
            return PretrainCIFAR100SplitDataModule(args)
        else:
            return PretrainCIFARDataModule(args)
    if mode == "pretrainSelfsupervised":
        if args.dataset == "ImageNet":
            return PretrainSelfsupervisedImageNetDataModule(args)
        elif "CIFAR100Split_" in args.dataset:
            raise NotImplementedError("No selfsupervised datamodule for CIFAR100Split_")
        else:
            return PretrainSelfsupervisedCIFARDataModule(args)
    elif mode == "discover":
        if args.dataset == "ImageNet":
            return DiscoverImageNetDataModule(args)
        elif "CIFAR100Split_" in args.dataset:
            return DiscoverCIFAR100SplitDataModule(args)
        else:
            return DiscoverCIFARDataModule(args)
    elif mode == "evaluate":
        if args.dataset == "ImageNet":
            return EvaluateImageNetDataModule(args)
        elif "CIFAR100Split_" in args.dataset:
            return EvaluateCIFAR100SplitDataModule(args)
        else:
            return EvaluateCIFARDataModule(args)


class PretrainCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes

        self.dataset_class = getattr(torchvision.datasets, args.dataset)

        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )
        train_indices_lab = np.where(
            np.isin(np.array(self.train_dataset.targets), labeled_classes)
        )[0]
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab)

        # val datasets
        self.val_dataset = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        val_indices_lab = np.where(np.isin(np.array(self.val_dataset.targets), labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=False,
            persistent_workers=use_persistent_workers
        )


class PretrainCIFAR100SplitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = CIFAR100
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)
        self.split = args.dataset[-4:]

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes, unlabeled_classes, target_transform = get_targets(self.split)

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train, target_transform=target_transform
        )
        train_indices_lab = np.where(
            np.isin(np.array(self.train_dataset.targets), labeled_classes)
        )[0]
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab)

        # val datasets
        self.val_dataset = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val, target_transform=target_transform
        )
        val_indices_lab = np.where(np.isin(np.array(self.val_dataset.targets), labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=False,
            persistent_workers=use_persistent_workers
        )


class PretrainSelfsupervisedCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.dataset_class = getattr(torchvision.datasets, args.dataset)

        self.unlabeled_data_only = args.unlabeled_data_only
        self.labeled_data_only = args.labeled_data_only
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes

        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            is_swav=True
        )
        self.transform_val = get_transforms(
            "eval",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            is_swav=True
        )

        # Needed for SwAV
        self.num_train_samples = 50000
        self.num_val_samples = 10000

        if self.labeled_data_only or self.unlabeled_data_only:
            numerator = self.num_labeled_classes if self.labeled_data_only else self.num_unlabeled_classes

            num_classes = self.num_labeled_classes + self.num_unlabeled_classes
            fraction = numerator / num_classes
            self.num_train_samples = int(self.num_train_samples * fraction)
            self.num_val_samples = int(self.num_val_samples * fraction)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if self.labeled_data_only or self.unlabeled_data_only:
            if self.labeled_data_only:
                target_classes = range(self.num_labeled_classes)
            else:
                target_classes = range(self.num_labeled_classes, self.num_labeled_classes + self.num_labeled_classes)

            target_mapping = {old: new for old, new in zip(target_classes, range(len(target_classes)))}
            def target_transform(y):
                return target_mapping[y]

            # train dataset
            train_dataset = self.dataset_class(
                self.data_dir, train=True, transform=self.transform_train, target_transform=target_transform
            )
            train_indices_unlab = np.where(
                np.isin(np.array(train_dataset.targets), target_classes)
            )[0]
            self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)

            # val datasets
            self.val_dataset = self.dataset_class(
                self.data_dir, train=False, transform=self.transform_val, target_transform=target_transform
            )
            val_indices_unlab = np.where(np.isin(np.array(self.val_dataset.targets), target_classes))[0]
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_unlab)
        else:
            # train dataset
            self.train_dataset = self.dataset_class(
                self.data_dir, train=True, transform=self.transform_train
            )

            # val datasets
            self.val_dataset = self.dataset_class(
                self.data_dir, train=False, transform=self.transform_val
            )

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=False,
            persistent_workers=use_persistent_workers
        )


class DiscoverCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.unlabeled_data_only = args.unlabeled_data_only

        self.dataset_class = getattr(torchvision.datasets, args.dataset)

        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )

        if self.unlabeled_data_only:
            target_mapping = {old: new for old, new in zip(unlabeled_classes, range(self.num_unlabeled_classes))}
            def target_transform(y):
                return target_mapping[y]
        else:
            def target_transform(y):
                return y

        # val datasets
        val_dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_val, target_transform=target_transform
        )
        val_dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val, target_transform=target_transform
        )
        # unlabeled classes, train set
        val_indices_unlab_train = np.where(
            np.isin(np.array(val_dataset_train.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)
        # unlabeled classes, test set
        val_indices_unlab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)
        # labeled classes, test set
        val_indices_lab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)

        train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train, target_transform=target_transform
        )

        if self.unlabeled_data_only:
            self.train_dataset = torch.utils.data.Subset(train_dataset, val_indices_unlab_train)
            self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test]
        else:
            self.train_dataset = train_dataset
            self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    @property
    def dataloader_mapping(self):
        if self.unlabeled_data_only:
            return {0: "unlab/train", 1: "unlab/test"}
        else:
            return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.val_datasets
        ]


class DiscoverCIFAR100SplitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes

        self.dataset_class = CIFAR100
        self.split = args.dataset[-4:]

        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes, unlabeled_classes, target_transform = get_targets(self.split)
        all_classes = labeled_classes + unlabeled_classes

        # val datasets
        val_dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_val, target_transform=target_transform
        )
        val_dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val, target_transform=target_transform
        )
        # unlabeled classes, train set
        val_indices_unlab_train = np.where(
            np.isin(np.array(val_dataset_train.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)
        # unlabeled classes, test set
        val_indices_unlab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)
        # labeled classes, test set
        val_indices_lab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)

        train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train, target_transform=target_transform
        )
        indices_train = np.where(
            np.isin(np.array(train_dataset.targets), all_classes)
        )[0]
        train_dataset = torch.utils.data.Subset(train_dataset, indices_train)

        self.train_dataset = train_dataset
        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.val_datasets
        ]


class EvaluateCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes

        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )

        # val datasets
        dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform
        )
        dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform
        )

        # labeled classes, train set
        indices_lab_train = np.where(
            np.isin(np.array(dataset_train.targets), labeled_classes)
        )[0]
        subset_lab_train = torch.utils.data.Subset(dataset_train, indices_lab_train)

        # unlabeled classes, train set
        indices_unlab_train = np.where(
            np.isin(np.array(dataset_train.targets), unlabeled_classes)
        )[0]
        subset_unlab_train = torch.utils.data.Subset(dataset_train, indices_unlab_train)

        self.train_datasets = [subset_lab_train, subset_unlab_train]

        # unlabeled classes, test set
        indices_unlab_test = np.where(
            np.isin(np.array(dataset_test.targets), unlabeled_classes)
        )[0]
        subset_unlab_test = torch.utils.data.Subset(dataset_test, indices_unlab_test)

        # labeled classes, test set
        indices_lab_test = np.where(
            np.isin(np.array(dataset_test.targets), labeled_classes)
        )[0]
        subset_lab_test = torch.utils.data.Subset(dataset_test, indices_lab_test)

        self.val_datasets = [subset_lab_test, subset_unlab_test]

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.train_datasets
        ]

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.val_datasets
        ]


class EvaluateCIFAR100SplitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes

        self.dataset_class = CIFAR100
        self.split = args.dataset[-4:]
        self.transform = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes, unlabeled_classes, target_transform = get_targets(self.split)

        # val datasets
        dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform, target_transform=target_transform
        )
        dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform, target_transform=target_transform
        )

        # labeled classes, train set
        indices_lab_train = np.where(
            np.isin(np.array(dataset_train.targets), labeled_classes)
        )[0]
        subset_lab_train = torch.utils.data.Subset(dataset_train, indices_lab_train)

        # unlabeled classes, train set
        indices_unlab_train = np.where(
            np.isin(np.array(dataset_train.targets), unlabeled_classes)
        )[0]
        subset_unlab_train = torch.utils.data.Subset(dataset_train, indices_unlab_train)

        self.train_datasets = [subset_lab_train, subset_unlab_train]

        # unlabeled classes, test set
        indices_unlab_test = np.where(
            np.isin(np.array(dataset_test.targets), unlabeled_classes)
        )[0]
        subset_unlab_test = torch.utils.data.Subset(dataset_test, indices_unlab_test)

        # labeled classes, test set
        indices_lab_test = np.where(
            np.isin(np.array(dataset_test.targets), labeled_classes)
        )[0]
        subset_lab_test = torch.utils.data.Subset(dataset_test, indices_lab_test)

        self.val_datasets = [subset_lab_test, subset_unlab_test]

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.train_datasets
        ]

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.val_datasets
        ]


class DiscoverDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return max([len(self.labeled_dataset), len(self.unlabeled_dataset)])

    def __getitem__(self, index):
        labeled_index = index % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_index]
        unlabeled_index = index % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_index]
        return (*labeled_data, *unlabeled_data)


class EvaluateImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.subset = args.imagenet_subset
        self.imagenet_split = args.imagenet_split
        self.dataset_class = ImageNetSubset
        self.transform = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        labeled_subset_train = self.dataset_class(train_data_dir, transform=self.transform, subset=self.subset,
                                                  subset_split=self.imagenet_split, partition="labeled")
        unlabeled_subset_train = self.dataset_class(train_data_dir, transform=self.transform, subset=self.subset,
                                                    subset_split=self.imagenet_split, partition="unlabeled")

        self.train_datasets = [labeled_subset_train, unlabeled_subset_train]

        # val datasets
        unlabeled_subset_test = self.dataset_class(val_data_dir, transform=self.transform, subset=self.subset,
                                                   subset_split=self.imagenet_split, partition="unlabeled")
        labeled_subset_test = self.dataset_class(val_data_dir, transform=self.transform, subset=self.subset,
                                                 subset_split=self.imagenet_split, partition="labeled")

        self.val_datasets = [labeled_subset_test, unlabeled_subset_test]

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.train_datasets
        ]

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.val_datasets
        ]


class DiscoverImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.subset = args.imagenet_subset
        self.imagenet_split = args.imagenet_split
        self.unlabeled_data_only = args.unlabeled_data_only
        self.dataset_class = ImageNetSubset
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        labeled_subset = self.dataset_class(train_data_dir, transform=self.transform_train,
                                            subset=self.subset, subset_split=self.imagenet_split,
                                            partition="labeled", unlabeled_only=self.unlabeled_data_only)
        unlabeled_subset = self.dataset_class(train_data_dir, transform=self.transform_train,
                                              subset=self.subset, subset_split=self.imagenet_split,
                                              partition="unlabeled", unlabeled_only=self.unlabeled_data_only)

        # val datasets
        unlabeled_subset_train = self.dataset_class(train_data_dir, transform=self.transform_val,
                                                    subset=self.subset, subset_split=self.imagenet_split,
                                                    partition="unlabeled", unlabeled_only=self.unlabeled_data_only)
        unlabeled_subset_test = self.dataset_class(val_data_dir, transform=self.transform_val,
                                                   subset=self.subset, subset_split=self.imagenet_split,
                                                   partition="unlabeled", unlabeled_only=self.unlabeled_data_only)
        labeled_subset_test = self.dataset_class(val_data_dir, transform=self.transform_val,
                                                 subset=self.subset, subset_split=self.imagenet_split,
                                                 partition="labeled", unlabeled_only=self.unlabeled_data_only)

        if self.unlabeled_data_only:
            self.train_dataset = unlabeled_subset
            self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test]
        else:
            self.train_dataset = DiscoverDataset(labeled_subset, unlabeled_subset)
            self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test, labeled_subset_test]

    @property
    def dataloader_mapping(self):
        if self.unlabeled_data_only:
            return {0: "unlab/train", 1: "unlab/test"}
        else:
            return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=not use_persistent_workers,
                drop_last=False,
                persistent_workers=use_persistent_workers
            )
            for dataset in self.val_datasets
        ]


class PretrainImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.subset = args.imagenet_subset
        self.imagenet_split = args.imagenet_split
        self.dataset_class = ImageNetSubset
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        labeled_subset = self.dataset_class(train_data_dir, transform=self.transform_train, subset=self.subset,
                                            subset_split=self.imagenet_split, partition="labeled")

        self.train_dataset = labeled_subset

        # val datasets
        labeled_subset_test = self.dataset_class(val_data_dir, transform=self.transform_val, subset=self.subset,
                                                 subset_split=self.imagenet_split, partition="labeled")

        self.val_dataset = labeled_subset_test

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=False,
            persistent_workers=use_persistent_workers
        )


class PretrainSelfsupervisedImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.subset = args.imagenet_subset
        self.imagenet_split = args.imagenet_split
        self.dataset_class = ImageNetSubset

        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            is_swav=True
        )
        self.transform_val = get_transforms(
            "eval",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            is_swav=True
        )

        # Contains the number of train and val samples for each split
        # This is needed for SwAV
        # Yes, it's ugly as hell
        num_samples = {
            "all": {
                "all": {
                    "A": (1167644, 45600),
                    "B": (1168232, 45600),
                    "C": (1168215, 45600)
                },
                "entity30": {
                    "l1u1": (153598, 6000),
                    "l2u1": (154261, 6000),
                    "l3u1": (153866, 6000),
                    "l1u2": (153567, 6000),
                    "l2u2": (154230, 6000),
                    "l3u2": (153835, 6000)
                }
            },
            "unlabeled": {
                "all": {
                    "A": (37885, 1500),
                    "B": (38473, 1500),
                    "C": (38456, 1500)
                },
                "entity30": {
                    "l1u1": (38689, 1500),
                    "l2u1": (38689, 1500),
                    "l3u1": (38689, 1500),
                    "l1u2": (38658, 1500),
                    "l2u2": (38658, 1500),
                    "l3u2": (38658, 1500)
                }
            },
            "labeled": {
                "all": {
                    "A": (1129759, 44100),
                    "B": (1129759, 44100),
                    "C": (1129759, 44100)
                },
                "entity30": {
                    "l1u1": (114909, 4500),
                    "l2u1": (115572, 4500),
                    "l3u1": (115177, 4500),
                    "l1u2": (114909, 4500),
                    "l2u2": (115572, 4500),
                    "l3u2": (115177, 4500)
                }
            }
        }

        if args.labeled_data_only:
            self.partition = "labeled"
        elif args.unlabeled_data_only:
            self.partition = "unlabeled"
        else:
            self.partition = "all"

        self.num_train_samples = num_samples[self.partition][self.subset][self.imagenet_split][0]
        self.num_val_samples = num_samples[self.partition][self.subset][self.imagenet_split][1]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        labeled_only = self.partition == "labeled"
        unlabeled_only = self.partition == "unlabeled"

        self.train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train, subset=self.subset,
                                                subset_split=self.imagenet_split, partition=self.partition,
                                                labeled_only=labeled_only, unlabeled_only=unlabeled_only)

        self.val_dataset = self.dataset_class(val_data_dir, transform=self.transform_val, subset=self.subset,
                                              subset_split=self.imagenet_split, partition=self.partition,
                                              labeled_only=labeled_only, unlabeled_only=unlabeled_only)

    def train_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=True,
            persistent_workers=use_persistent_workers
        )

    def val_dataloader(self):
        use_persistent_workers = self.num_workers > 0

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=not use_persistent_workers,
            drop_last=False,
            persistent_workers=use_persistent_workers
        )
