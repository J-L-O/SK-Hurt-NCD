#!/usr/bin/env bash
# This script is used to pretrain a model on our CIFAR100-based benchmark using supervised learning.
# Usage: pretrain_cifar100_supervised.sh <labeled_set>

labeled_set="$1"

if [ -z "$1" ]
  then
    echo "No labeled set supplied"
    exit 1
fi

# To lowercase
labeled_set=${labeled_set,,}

python main_pretrain.py  \
  --gpus 1 \
  --num_workers "$NUM_WORKERS" \
  --precision 16 \
  --dataset CIFAR100Split_"$labeled_set"u1 \
  --data_dir "$CIFAR100_PATH" \
  --max_epochs 200 \
  --batch_size 256 \
  --num_labeled_classes 40 \
  --num_unlabeled_classes 10 \
  --comment standard \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --log_dir "$LOG_PATH"
