#!/usr/bin/env bash
# This script is used to discover novel classes in our CIFAR100-based benchmark based on supervised pretraining.
# Usage: discover_cifar100_supervised.sh <labeled_set> <unlabeled_set> <alpha>

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
  --dataset CIFAR100Split_"$labeled_set""$unlabeled_set" \
  --gpus 1 \
  --precision 16 \
  --data_dir "$CIFAR100_PATH" \
  --max_epochs 500 \
  --batch_size 512 \
  --num_labeled_classes 40 \
  --num_unlabeled_classes 10 \
  --pretrained "$CHECKPOINT_PATH"/pretrain-resnet18-CIFAR100Split_"$labeled_set"U1-standard.cp \
  --comment standard_alpha"$alpha" \
  --multicrop \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --checkpoint_freq 500 \
  --log_dir "$LOG_PATH"  \
  --num_workers "$NUM_WORKERS" \
  --mix_sup_selfsup \
  --mix_sup_weight "$alpha"
