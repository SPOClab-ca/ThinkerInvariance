#!/bin/bash

DATASET="MMI"
BATCH_SIZE=16
EPOCHS=30
EWMA=5
MIXUP=8.0
RESULTS_PREFIX="results/MMI/6s/lmso"

# Referenced wrt toplevel project directory to ensure experiments are run from there

# TIDNet
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-2-TIDNet" "TIDNet --tlen 6" --targets 2
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-3-TIDNet" "TIDNet --tlen 6" --targets 3
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-4-TIDNet" "TIDNet --tlen 6" --targets 4

# EEGNet
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-2-EEGNet" "EEGNet --tlen 6" --targets 2
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-3-EEGNet" "EEGNet --tlen 6" --targets 3
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP "${RESULTS_PREFIX}-4-EEGNet" "EEGNet --tlen 6" --targets 4
