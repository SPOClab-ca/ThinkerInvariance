#!/bin/bash

USAGE_STRING="Usage: _tidnet_basic {dataset} {batch_size} {epochs} {ewma_model} {mixup} {results_prefix} {*model} +{*dataset args}"

if [ $# -lt 5 ]; then
  echo "Not enough arguments provided"
  echo $USAGE_STRING
  return
elif [ $# -ge 7 ]; then
  MODEL=$7
else
  MODEL="TIDNet"
fi

DATASET=$1
BATCH_SIZE=$2
EPOCHS=$3
WARMUP=$(( EPOCHS / 5 ))
EWMA=$4
MIXUP=$5
RESULTS_PREFIX=$6

parentdir="$(dirname "$RESULTS_PREFIX")"
echo "Saving results to $parentdir"
mkdir -p $parentdir

# LO(/M)SO
python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup $MIXUP --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_ea_mixup.xlsx" $DATASET "${@:8}"

python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup 0 --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_ea.xlsx" $DATASET "${@:8}"

python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup $MIXUP --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_mixup.xlsx" --no-alignment $DATASET "${@:8}"

python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup 0 --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}.xlsx" --no-alignment $DATASET "${@:8}"

# MDL experiments
python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup $MIXUP --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_ea_mixup_mdl.xlsx" --use-training $DATASET "${@:8}"

python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup 0 --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_ea_mdl.xlsx" --use-training $DATASET "${@:8}"

python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup $MIXUP --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_mixup_mdl.xlsx" --no-alignment --use-training $DATASET "${@:8}"

python3 main.py $MODEL -wu $WARMUP -e $EPOCHS -bs $BATCH_SIZE --mixup 0 --ewma-model $EWMA --save-params \
        --results "${RESULTS_PREFIX}_mdl.xlsx" --no-alignment --use-training $DATASET "${@:8}"