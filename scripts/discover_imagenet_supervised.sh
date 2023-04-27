#!/usr/bin/env bash
# This script is used to discover novel classes in our ImageNet-based benchmark based on supervised pretraining.
# Usage: discover_imagenet_supervised.sh <labeled_set> <unlabeled_set> <alpha>

labeled_set="$1"
unlabeled_set="$2"
alpha="$3"

if [ -z "$1" ]
  then
    echo "No labeled set supplied"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No unlabeled set supplied"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No alpha supplied"
    exit 1
fi

# To lowercase
labeled_set=${labeled_set,,}
unlabeled_set=${unlabeled_set,,}

python main_discover.py \
  --dataset ImageNet \
  --gpus 1 \
  --precision 16 \
  --data_dir "$IMAGENET_PATH" \
  --max_epochs 60 \
  --base_lr 0.2 \
  --warmup_epochs 5 \
  --batch_size 512 \
  --num_labeled_classes 90 \
  --num_unlabeled_classes 30 \
  --num_heads 4 \
  --pretrained "$CHECKPOINT_PATH"/pretrain-resnet18-ImageNet-entity30_"$labeled_set".cp \
  --imagenet_subset entity30 \
  --imagenet_split "$labeled_set""$unlabeled_set" \
  --comment entity30_"$labeled_set""$unlabeled_set"_alpha"$alpha" \
  --overcluster_factor 4 \
  --multicrop \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --checkpoint_freq 60 \
  --log_dir "$LOG_PATH" \
  --num_workers "$NUM_WORKERS" \
  --mix_sup_selfsup \
  --mix_sup_weight "$alpha"
