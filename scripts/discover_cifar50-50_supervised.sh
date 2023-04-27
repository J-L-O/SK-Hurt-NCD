#!/usr/bin/env bash
# This script is used to discover novel classes in our CIFAR100-based benchmark based on supervised pretraining.
# Usage: discover_cifar100_supervised.sh <alpha>

alpha="$1"

if [ -z "$1" ]
  then
    echo "No alpha supplied"
    exit 1
fi

python main_discover.py \
  --dataset CIFAR100 \
  --gpus 1 \
  --precision 16 \
  --data_dir "$CIFAR100_PATH" \
  --max_epochs 500 \
  --batch_size 512 \
  --num_labeled_classes 50 \
  --num_unlabeled_classes 50 \
  --pretrained "$CHECKPOINT_PATH"/pretrain-resnet18-CIFAR100-50-50_swav.cp \
  --comment 50-50_swav_mix_alpha"$alpha" \
  --multicrop \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --checkpoint_freq 500 \
  --log_dir "$LOG_PATH"  \
  --num_workers "$NUM_WORKERS" \
  --from_swav \
  --mix_sup_selfsup \
  --mix_sup_weight "$alpha"
