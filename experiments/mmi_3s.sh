#!/bin/bash

DATASET="MMI"
BATCH_SIZE=16
EPOCHS=30
EWMA=5
MIXUP=8.0
RESULTS_PREFIX="results/MMI/3s/lmso"

# TIDNet
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-2-TIDNet" "TIDNet" --targets 2
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-3-TIDNet" "TIDNet" --targets 3
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-4-TIDNet" "TIDNet" --targets 4

EPOCHS=10
EWMA=2

# Shallow
#./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-2-Dose" "Dose" --targets 2
#./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-3-Dose" "Dose" --targets 3
#./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-4-Dose" "Dose" --targets 4

EPOCHS=30
EWMA=5

# EEGNet
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-2-EEGNet" "EEGNet" --targets 2
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-3-EEGNet" "EEGNet" --targets 3
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-4-EEGNet" "EEGNet" --targets 4
