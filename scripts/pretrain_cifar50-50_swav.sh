#!/usr/bin/env bash
# This script is used to pretrain a model on CIFAR100-50-50 using SwAV.
# Usage: pretrain_cifar100_swav.sh

python main_swav.py  \
  --gpus 1 \
  --accelerator ddp \
  --arch resnet18 \
  --hidden_mlp 512 \
  --num_workers "$NUM_WORKERS" \
  --precision 16 \
  --dataset CIFAR100 \
  --data_dir "$CIFAR100_PATH" \
  --num_labeled_classes 50 \
  --num_unlabeled_classes 50 \
  --comment 50-50_swav \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --log_dir "$LOG_PATH" \
  --multicrop \
  --nmb_prototypes 3000 \
  --max_epochs 800 \
  --warmup_epochs 0 \
  --batch_size 256 \
  --queue_length 3840 \
  --epoch_queue_starts 100 \
  --optimizer sgd \
  --lars_wrapper \
  --learning_rate 0.6 \
  --final_lr 0.0006 \
  --freeze_prototypes_epochs 1
