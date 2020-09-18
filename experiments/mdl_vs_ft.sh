#!/bin/bash

TRIAL_LEN=3

function fine_tune() {
    PARAMS=$1
    RESULTS=$2
    DATASET=$3
    python3 main.py TIDNet -bs 2 -wu 0 -e 10 --ewma-model 3 --load-params $PARAMS --results $RESULTS --tlen $TRIAL_LEN \
            --save-params --fine-tune $DATASET
}

function targetted_mdl() {
    BATCH_SIZE=$1
    EPOCHS=$2
    WARMUP=$(( EPOCHS / 5 ))
    EWMA_MODEL=$3
    RESULTS=$4
    DATASET=$5
    python3 main.py TIDNet -bs $BATCH_SIZE -wu $WARMUP -e $EPOCHS --ewma-model $EWMA_MODEL --tlen $TRIAL_LEN \
            --results $RESULTS --save-params --mdl-hold $DATASET
}

# Fine Tuning
# MMI
TRIAL_LEN=3
TRIAL_LEN=6
fine_tune "saved_models/MMI/${TRIAL_LEN}s/lmso-2-TIDNet/" "results/MMI/${TRIAL_LEN}s/fine-tuned-2.xlsx" "--no-alignment MMI --targets 2"
fine_tune "saved_models/MMI/${TRIAL_LEN}s/lmso-3-TIDNet_ea/" "results/MMI/${TRIAL_LEN}s/fine-tuned-3.xlsx" "MMI --targets 3"
fine_tune "saved_models/MMI/${TRIAL_LEN}s/lmso-4-TIDNet_ea/" "results/MMI/${TRIAL_LEN}s/fine-tuned-4.xlsx" "MMI --targets 4"

# ERN
TRIAL_LEN=2
fine_tune "saved_models/ERN/ERN/loso_ea_mixup/" "results/eog_free/ERN/fine-tuned.xlsx" "ERN"


# Targetted MDL
# MMI
#TRIAL_LEN=3
#TRIAL_LEN=6
#targetted_mdl 16 30 "5 --no-alignment" "results/MMI/${TRIAL_LEN}s/targetted_mdl_2.xlsx" "MMI --targets 2"
#targetted_mdl 16 30 5 "results/MMI/${TRIAL_LEN}s/targetted_mdl_3.xlsx" "MMI --targets 3"
#targetted_mdl 16 30 5 "results/MMI/${TRIAL_LEN}s/targetted_mdl_4.xlsx" "MMI --targets 4"

#ERN
TRIAL_LEN=2
targetted_mdl 4 20 15 "results/eog_free/ERN/targetted_mdl.xlsx" "ERN"
