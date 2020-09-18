#!/bin/bash

DATASET="BCI"
BATCH_SIZE=60
EPOCHS=100
EWMA=5
MIXUP=2.0
RESULTS_TOPLEVEL="results"

function run_iv_2a() {
    RESULTS_SUFFIX=$1
    MODEL=$2

    RESULTS_PREFIX="$RESULTS_TOPLEVEL/eog_free/$RESULTS_SUFFIX"
    ./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP $RESULTS_PREFIX $MODEL

    RESULTS_PREFIX="$RESULTS_TOPLEVEL/all/$RESULTS_SUFFIX"
    ./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP $RESULTS_PREFIX "$MODEL --include-eog"

    RESULTS_PREFIX="$RESULTS_TOPLEVEL/eog_ica/$RESULTS_SUFFIX"
    ./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP $RESULTS_PREFIX "$MODEL --ica-eog"
}

run_iv_2a BCI_2a/loso-TIDNet TIDNet

run_iv_2a BCI_2a/loso-EEGNet EEGNet
