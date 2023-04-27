#!/usr/bin/env bash
# This script is used to pretrain a model on our CIFAR100-based benchmark based on SwAV pretraining.
# Usage: pretrain_cifar100_swav.sh <alpha>

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
  --data_dir /dataset/ \
  --max_epochs 500 \
  --batch_size 512 \
  --num_labeled_classes "$labeled_classes" \
  --num_unlabeled_classes "$unlabeled_classes" \
  --pretrained /checkpoints/pretrain-resnet18-CIFAR100-"$labeled_classes"-"$unlabeled_classes"_swav.cp \
  --comment "$labeled_classes"-"$unlabeled_classes"_swav_unlabeled \
  --multicrop \
  --checkpoint_dir /checkpoints/ \
  --checkpoint_freq 500 \
  --log_dir /logs/  \
  --num_workers "$num_workers" \
  --from_swav \
  --unsupervised
