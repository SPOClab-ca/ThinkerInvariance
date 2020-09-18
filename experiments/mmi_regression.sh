#!/bin/bash

NUM_POINTS=20
XVAL_FOLDS=10

BATCH_SIZE=16
# No EWMA model parameters are used in these tests to avoid the overfitting that can sometimes happen with few subjects
# Instead, the epoch with the best validation performance is used
RESULTS_PREFIX="results/MMI/subject_regression/"

function target_run() {
    TARGETS=$1
    MODEL=$2
    python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --no-alignment --tmin 0 --tlen 3  \
            --results "${RESULTS_PREFIX}${MODEL}_${TARGETS}.xlsx" MMI --targets $TARGETS  \
            --xval-folds $XVAL_FOLDS --subj-subsets $NUM_POINTS

    python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --tmin 0 --tlen 3  \
            --results "${RESULTS_PREFIX}${MODEL}_${TARGETS}_ea.xlsx" MMI --targets $TARGETS --xval-folds $XVAL_FOLDS \
            --subj-subsets $NUM_POINTS
}

function model_run() {
    MODEL=$1
    target_run 2 $MODEL
    target_run 3 $MODEL
    target_run 4 $MODEL
}

EPOCHS=30
WARMUP=6
model_run TIDNet

model_run EEGNet

#EPOCHS=30
#WARMUP=
#BATCH_SIZE=16
#model_run Dose

