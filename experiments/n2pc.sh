#!/bin/bash

DATASET="N2PC"
BATCH_SIZE=16
EPOCHS=20
EWMA=3
MIXUP=8.0
RESULTS_PREFIX="results/N2PC/loso"

# Referenced wrt toplevel project directory to ensure experiments are run from there
./experiments/_tidnet_basic.sh $DATASET $BATCH_SIZE $EPOCHS $EWMA $MIXUP $RESULTS_PREFIX
