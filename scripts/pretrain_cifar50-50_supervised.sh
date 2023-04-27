#!/usr/bin/env bash
# This script is used to pretrain a model on CIFAR100-50-50 using supervised learning.
# Usage: pretrain_cifar100_supervised.sh

python main_pretrain.py  \
  --gpus 1 \
  --num_workers "$NUM_WORKERS" \
  --precision 16 \
  --dataset CIFAR100 \
  --data_dir "$CIFAR100_PATH" \
  --max_epochs 200 \
  --batch_size 256 \
  --num_labeled_classes 50 \
  --num_unlabeled_classes 50 \
  --comment 50-50_standard \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --log_dir "$LOG_PATH"
