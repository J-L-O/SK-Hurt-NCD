from typing import Tuple, List, Dict

import numpy as np
from torchvision.datasets import ImageFolder

from utils.transforms import DiscoverTargetTransform

IMAGENET_CLASSES_118 = [
    "n01498041",
    "n01537544",
    "n01580077",
    "n01592084",
    "n01632777",
    "n01644373",
    "n01665541",
    "n01675722",
    "n01688243",
    "n01729977",
    "n01775062",
    "n01818515",
    "n01843383",
    "n01883070",
    "n01950731",
    "n02002724",
    "n02013706",
    "n02092339",
    "n02093256",
    "n02095314",
    "n02097130",
    "n02097298",
    "n02098413",
    "n02101388",
    "n02106382",
    "n02108089",
    "n02110063",
    "n02111129",
    "n02111500",
    "n02112350",
    "n02115913",
    "n02117135",
    "n02120505",
    "n02123045",
    "n02125311",
    "n02134084",
    "n02167151",
    "n02190166",
    "n02206856",
    "n02231487",
    "n02256656",
    "n02398521",
    "n02480855",
    "n02481823",
    "n02490219",
    "n02607072",
    "n02666196",
    "n02672831",
    "n02704792",
    "n02708093",
    "n02814533",
    "n02817516",
    "n02840245",
    "n02843684",
    "n02870880",
    "n02877765",
    "n02966193",
    "n03016953",
    "n03017168",
    "n03026506",
    "n03047690",
    "n03095699",
    "n03134739",
    "n03179701",
    "n03255030",
    "n03388183",
    "n03394916",
    "n03424325",
    "n03467068",
    "n03476684",
    "n03483316",
    "n03627232",
    "n03658185",
    "n03710193",
    "n03721384",
    "n03733131",
    "n03785016",
    "n03786901",
    "n03792972",
    "n03794056",
    "n03832673",
    "n03843555",
    "n03877472",
    "n03899768",
    "n03930313",
    "n03935335",
    "n03954731",
    "n03995372",
    "n04004767",
    "n04037443",
    "n04065272",
    "n04069434",
    "n04090263",
    "n04118538",
    "n04120489",
    "n04141975",
    "n04152593",
    "n04154565",
    "n04204347",
    "n04208210",
    "n04209133",
    "n04258138",
    "n04311004",
    "n04326547",
    "n04367480",
    "n04447861",
    "n04483307",
    "n04522168",
    "n04548280",
    "n04554684",
    "n04597913",
    "n04612504",
    "n07695742",
    "n07697313",
    "n07697537",
    "n07716906",
    "n12998815",
    "n13133613",
]

IMAGENET_CLASSES_30 = {
    "A": [
        "n01580077",
        "n01688243",
        "n01883070",
        "n02092339",
        "n02095314",
        "n02098413",
        "n02108089",
        "n02120505",
        "n02123045",
        "n02256656",
        "n02607072",
        "n02814533",
        "n02840245",
        "n02843684",
        "n02877765",
        "n03179701",
        "n03424325",
        "n03483316",
        "n03627232",
        "n03658185",
        "n03785016",
        "n03794056",
        "n03899768",
        "n04037443",
        "n04069434",
        "n04118538",
        "n04154565",
        "n04311004",
        "n04522168",
        "n07695742",
    ],
    "B": [
        "n01883070",
        "n02013706",
        "n02093256",
        "n02097130",
        "n02101388",
        "n02106382",
        "n02112350",
        "n02167151",
        "n02490219",
        "n02814533",
        "n02843684",
        "n02870880",
        "n03017168",
        "n03047690",
        "n03134739",
        "n03394916",
        "n03424325",
        "n03483316",
        "n03658185",
        "n03721384",
        "n03733131",
        "n03786901",
        "n03843555",
        "n04120489",
        "n04152593",
        "n04208210",
        "n04258138",
        "n04522168",
        "n04554684",
        "n12998815",
    ],
    "C": [
        "n01580077",
        "n01592084",
        "n01632777",
        "n01775062",
        "n01818515",
        "n02097130",
        "n02097298",
        "n02098413",
        "n02111500",
        "n02115913",
        "n02117135",
        "n02398521",
        "n02480855",
        "n02817516",
        "n02843684",
        "n02877765",
        "n02966193",
        "n03095699",
        "n03394916",
        "n03424325",
        "n03710193",
        "n03733131",
        "n03785016",
        "n03995372",
        "n04090263",
        "n04120489",
        "n04326547",
        "n04522168",
        "n07697537",
        "n07716906",
    ],
}


def _get_class_list_path(is_labeled, id):
    labeled = "labeled" if is_labeled else "unlabeled"
    return f"./utils/imagenet_domain_gap/imagenet_{labeled}_{id}.txt"


class ImageNetSubset(ImageFolder):
    SUPPORTED_SPLITS = {
        "all": ["A", "B", "C"],
        "entity30": ["l1u1", "l1u2", "l2u1", "l2u2", "l3u1", "l3u2"]
        }

    def __init__(
            self,
            *args,
            subset="all",
            subset_split="A",
            partition="labeled",
            labeled_only=False,
            unlabeled_only=False,
            **kwargs
    ):
        assert subset in self.SUPPORTED_SPLITS.keys(), f"Subset {subset} is not supported!"
        assert subset_split in self.SUPPORTED_SPLITS[subset], f"Split {subset_split} is not supported for subset {subset}!"

        self.subset = subset
        self.subset_split = subset_split
        self.partition = partition
        self.labeled_only = labeled_only
        self.unlabeled_only = unlabeled_only
        super(ImageNetSubset, self).__init__(*args, **kwargs)

    def _initialize_entity30(self, class_to_idx):
        labeled_split = int(self.subset_split[1])
        unlabeled_split = int(self.subset_split[3])
        labeled_classes = list(np.loadtxt(_get_class_list_path(True, labeled_split), dtype=str))
        unlabeled_classes = list(np.loadtxt(_get_class_list_path(False, unlabeled_split), dtype=str))

        labeled_class_idxs = np.array([class_to_idx[name] for name in labeled_classes])
        unlabeled_class_idxs = np.array([class_to_idx[name] for name in unlabeled_classes])

        # target transform
        if self.labeled_only:
            all_class_idxs = labeled_class_idxs
        elif self.unlabeled_only:
            all_class_idxs = unlabeled_class_idxs
        else:
            all_class_idxs = np.concatenate((labeled_class_idxs, unlabeled_class_idxs))

        target_transform = DiscoverTargetTransform(
            {original: target for target, original in enumerate(all_class_idxs)}
        )
        self.target_transform = target_transform

        if self.partition == "labeled":
            new_classes = labeled_classes
            new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, labeled_class_idxs)}
        elif self.partition == "unlabeled":
            new_classes = unlabeled_classes
            new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, unlabeled_class_idxs)}
        else:
            new_classes = labeled_classes + unlabeled_classes
            new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, all_class_idxs)}

        return new_classes, new_class_to_idx

    def _initialize_all(self, class_to_idx):
        mapping = {c[:9]: i for c, i in class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]
        unlabeled_classes = IMAGENET_CLASSES_30[self.subset_split]
        unlabeled_classes.sort()
        unlabeled_class_idxs = [mapping[c] for c in unlabeled_classes]

        # target transform
        if self.labeled_only:
            all_classes = labeled_classes
        elif self.unlabeled_only:
            all_classes = unlabeled_classes
        else:
            all_classes = labeled_classes + unlabeled_classes

        target_transform = DiscoverTargetTransform(
            {mapping[c]: i for i, c in enumerate(all_classes)}
        )
        self.target_transform = target_transform

        if self.partition == "labeled":
            new_classes = labeled_classes
            new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, labeled_class_idxs)}
        elif self.partition == "unlabeled":
            new_classes = unlabeled_classes
            new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, unlabeled_class_idxs)}
        else:
            new_classes = all_classes
            all_class_idxs = np.concatenate((labeled_class_idxs, unlabeled_class_idxs))
            new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, all_class_idxs)}

        return new_classes, new_class_to_idx

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes, class_to_idx = super(ImageNetSubset, self)._find_classes(dir)

        if self.subset == "all":
            subset_classes, subset_class_to_idx = self._initialize_all(class_to_idx)
        else:
            subset_classes, subset_class_to_idx = self._initialize_entity30(class_to_idx)

        return subset_classes, subset_class_to_idx
