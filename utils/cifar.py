from typing import Callable, Tuple, List

all_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
    'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
    'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
    'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
    'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster',
    'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm'
]

labeled_list1 = [
    'dolphin', 'otter', 'seal', 'whale',
    'flatfish', 'ray', 'shark', 'trout',
    'poppy', 'rose', 'sunflower', 'tulip',
    'bowl', 'can', 'cup', 'plate',
    'mushroom', 'orange', 'pear', 'sweet_pepper',
    'keyboard', 'lamp', 'telephone', 'television',
    'chair', 'couch', 'table', 'wardrobe',
    'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'leopard', 'lion', 'tiger', 'wolf',
    'castle', 'house', 'road', 'skyscraper'
]

unlabeled_list1 = [
    'beaver',
    'aquarium_fish',
    'orchid',
    'bottle',
    'apple',
    'clock',
    'bed',
    'bee',
    'bear',
    'bridge'
]

labeled_list2 = [
    'forest', 'mountain', 'plain', 'sea',
    'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'porcupine', 'possum', 'raccoon', 'skunk',
    'lobster', 'snail', 'spider', 'worm',
    'boy', 'girl', 'man', 'woman',
    'dinosaur', 'lizard', 'snake', 'turtle',
    'mouse', 'rabbit', 'shrew', 'squirrel',
    'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
    'bus', 'motorcycle', 'pickup_truck', 'train',
    'rocket', 'streetcar', 'tank', 'tractor'
]

unlabeled_list2 = [
    'cloud',
    'camel',
    'fox',
    'crab',
    'baby',
    'crocodile',
    'hamster',
    'maple_tree',
    'bicycle',
    'lawn_mower'
]


def get_targets(split: str) -> Tuple[List[int], List[int], Callable[[int], int]]:
    labeled_split, unlabeled_split = split[:2].upper(), split[2:].upper()

    labeled_options = ['L1', 'L2']
    unlabeled_options = ['U1', 'U2']

    assert labeled_split in labeled_options, f'Labeled split must be one of {labeled_options}'
    assert unlabeled_split in unlabeled_options, f'Unlabeled split must be one of {unlabeled_options}'

    class_to_idx = {_class: i for i, _class in enumerate(all_classes)}

    if labeled_split == 'L1':
        labeled_indices = [class_to_idx[cls_name] for cls_name in labeled_list1]
    else:
        labeled_indices = [class_to_idx[cls_name] for cls_name in labeled_list2]

    if unlabeled_split == 'U1':
        unlabeled_indices = [class_to_idx[cls_name] for cls_name in unlabeled_list1]
    else:
        unlabeled_indices = [class_to_idx[cls_name] for cls_name in unlabeled_list2]

    all_indices = labeled_indices + unlabeled_indices
    target_dict = {cls: mapped_cls for cls, mapped_cls in zip(all_indices, range(len(all_indices)))}

    def target_transform(cls: int) -> int:
        return target_dict[cls]

    return labeled_indices, unlabeled_indices, target_transform
