#!/usr/bin/env bash
# This script is used to pretrain a model on our ImageNet-based benchmark using supervised learning.
# Usage: pretrain_imagenet_supervised.sh <labeled_set>

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
  --dataset ImageNet \
  --data_dir "$IMAGENET_PATH" \
  --max_epochs 100 \
  --warmup_epochs 5 \
  --batch_size 512 \
  --num_labeled_classes 90 \
  --num_unlabeled_classes 30 \
  --comment entity30_"$labeled_set" \
  --checkpoint_dir "$CHECKPOINT_PATH" \
  --log_dir "$LOG_PATH" \
  --imagenet_subset entity30 \
  --imagenet_split "$labeled_set"u1  # Unlabeled set doesn't matter here, but we need to specify something
