#!/usr/bin/env bash
# This script is used to pretrain a model on our ImageNet-based benchmark using SwAV.
# Usage: pretrain_imagenet_swav.sh <labeled_set> <unlabeled_set>

labeled_set="$1"
unlabeled_set="$2"

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

# To lowercase
labeled_set=${labeled_set,,}
unlabeled_set=${unlabeled_set,,}

python main_swav.py  \
  --gpus 1 \
  --accelerator ddp \
  --arch resnet18 \
  --hidden_mlp 512 \
  --num_workers "$NUM_WORKERS" \
  --precision 16 \
  --dataset ImageNet \
  --data_dir "$IMAGENET_PATH" \
  --num_labeled_classes 90 \
  --num_unlabeled_classes 30 \
  --comment imagenet_swav_"$labeled_set""$unlabeled_set" \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --log_dir "$LOG_PATH" \
  --imagenet_subset entity30 \
  --imagenet_split "$labeled_set""$unlabeled_set" \
  --multicrop \
  --nmb_prototypes 3000 \
  --max_epochs 800 \
  --warmup_epochs 0 \
  --batch_size 256 \
  --queue_length 3840 \
  --epoch_queue_starts 60 \
  --optimizer sgd \
  --lars_wrapper \
  --learning_rate 0.6 \
  --final_lr 0.0006 \
  --freeze_prototypes_epochs 1
