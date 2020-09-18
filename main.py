import models
from utils import *

from datasets.bci_iv import bci_competitions
from datasets.mmi_physionet import mmidb, SUBJECTS_EEGBCI
from datasets.n2pc import n2pc, SUBJECTS_N2PC
from datasets.p300spell import p300_speller_system, SUBJECTS_SPELLER
from datasets.ern import ern, SUBJECTS_ERN

import mne
import argparse
from pathlib import Path


if __name__ == '__main__':

    # -----------------------------------------------------
    parser = argparse.ArgumentParser(description="Thinker invariance experiments. For N subjects models are trained "
                                                 "with N-2 subject's data, being validated against one leftover, "
                                                 "then tested on the final held out subject.")
    parser.add_argument('model', choices=models.MODELS.keys(), help='Which model to use.')
    parser.add_argument('--results', help='Where to store all the performance outputs (accuracy only).')

    # Model Saving/Loading
    parser.add_argument('--save-params', help='Save the best model of each trained to the saved_models/ directory. '
                                              'If a results file is specified, will organize accordingly.',
                        action='store_true')
    parser.add_argument('--load-params', help='Path to pre-trained model parameters. If fine-tune also selected, '
                                              'directory with models for all folds.')
    parser.add_argument('--fine-tune', help='Ignored without load-params, fine-tunes models in --load-params directory '
                                            'for each subject in dataset.', action='store_true')
    parser.add_argument('--mdl-hold', action='store_true', help='Hold out subjects during MDL training.')


    # Dataset Epoching
    parser.add_argument('--tmin', type=float, help='Start time in seconds with respect to event marker.')
    parser.add_argument('--tlen', type=float, help='The length of cropped events in seconds.')
    parser.add_argument('--tcrop', type=float, help='The length of sub-crops, None means no cropping')
    parser.add_argument('--teval', type=float, help='The evaluation offset w.r.t epoch marker if cropping')
    parser.add_argument('--crop-p', type=float, help='The probability of a random crop rather than the evaluation '
                                                     'offset.', default=0.5)

    # Different datasets and handlers
    subparsers = parser.add_subparsers(title='Datasets', description='Which dataset to load.', dest='dataset')
    bci_iv_parser = subparsers.add_parser('BCI', help='Use the BCI competition IV dataset 2a Motor Imagery.')
    bci_iv_parser.add_argument('--subject', choices=list(range(1, 10)), help='Train only a single subject', type=int)
    bci_iv_parser.add_argument('--rand-val', action='store_true', help='Rather than validate on held-out subjects, '
                                                                       'validate on a random portion of all training.')
    bci_iv_parser.add_argument('--ewma-normalize', action='store_true', help='EWMA normalization of entire recording '
                                                                             'before being epoching, as in prior work')
    bci_iv_parser.add_argument('--filter', action='store_true', help='4-40Hz filtering of entire recording '
                                                                     'before being epoching, as in prior work')
    bci_iv_parser.set_defaults(func=bci_competitions)

    mmidb_parser = subparsers.add_parser('MMI', help='Use the PhysioBank Movement/Motor Imagery Database')
    mmidb_parser.add_argument('--targets', type=int, choices=[2, 3, 4], help='Number of targets to classify', default=4)
    mmidb_parser.add_argument('--xval-folds', type=int, help='Number of cross-validation folds', default=5)
    mmidb_parser.add_argument('--subject', choices=SUBJECTS_EEGBCI, help='Train only a single subject', type=int)
    mmidb_parser.add_argument('--subj-subsets', help='Number of points to take of subsets of subjects.', type=int)
    mmidb_parser.set_defaults(func=mmidb)

    n2pc_parser = subparsers.add_parser('N2PC', help='Use one of the N2PC RSVP dataset.')
    n2pc_parser.add_argument('--subject', choices=SUBJECTS_N2PC, help='Train only a single subject')
    n2pc_parser.set_defaults(func=n2pc)

    ern_parser = subparsers.add_parser('ERN', help='Use of the INRIA ERN dataset.')
    ern_parser.add_argument('--subject', choices=SUBJECTS_ERN, help='Train only a single subject')
    ern_parser.add_argument('--xval-folds', type=int, help='Number of cross-validation folds', default=4)
    ern_parser.add_argument('--filter', action='store_true', help='Filter between 1-40 Hz to reproduce previous works.')
    ern_parser.set_defaults(func=ern)

    p300_parser = subparsers.add_parser('P300', help='Use P300 speller dataset.')
    p300_parser.add_argument('--subject', choices=SUBJECTS_SPELLER, help='Train only a single subject')
    p300_parser.add_argument('--decimate', type=int, default=2, help='Due to the high sampling rate of this dataset, '
                                                                     'it is prudent to decimate (select every Nth '
                                                                     'sample).')
    p300_parser.add_argument('--map-2d', action='store_true', help='Map onto a 2D grid.')
    p300_parser.add_argument('--drop-ref', action='store_true', help="Drop the A1 and A2 reference channels before "
                                                                     "classification.")
    p300_parser.set_defaults(func=p300_speller_system)

    # Training Options
    parser.add_argument('--subject-specific', action='store_true', help='Trains a model for each subject only using '
                                                                        'their own data; no other subject data.')
    parser.add_argument('--use-training', action='store_true', help='Also include the target subjects training data '
                                                                    'with the other subject data.')
    parser.add_argument('--val-out', action='store_true', help='Save the best_val.npy file containing '
                                                               'predicted and true validation labels.')
    parser.add_argument('--hide-test', action='store_true', help='Set this while tuning to not see test performance '
                                                                 'and focus only on validation results.')
    parser.add_argument('--full-test', action='store_true', help='Multiple runs, effectively nested cross-validation')
    parser.add_argument('--ewma-model', type=int, default=None, help='Rather than use the best validation model, take '
                                                                     'an exponentially weighted moving average of the '
                                                                     'last (this arg) epochs.')
    parser.add_argument('--include-eog', action='store_true', help="Use the EOG channels for classification.")
    parser.add_argument('--ica-eog', action='store_true', help="Run ICA and drop highly correlated components to EOG "
                                                               "channels before training.")

    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--warmup', '-wu', type=int, default=20, help='Number of epochs to ramp up the learning rate '
                                                                      'before decaying.')
    parser.add_argument('--warmup-drop', type=float, default=1.0, help='Factor to modify the learning rate after the '
                                                                       'warmup stage.')
    parser.add_argument('--epochs', '-e', help='Number of epochs per run.', type=int, default=100)
    parser.add_argument('--batch-size', '-bs', type=int, default=60)
    parser.add_argument('--normalizer', '-n', choices=NORMALIZERS.keys(), default='fixedscale')

    parser.add_argument('--label-smoothing', '-ls', help='How much to smooth the labels. 0:no smoothing, 1:full smooth',
                        type=float, default=0.2)
    parser.add_argument('--mixup', type=float, default=0, help='Amount of mixup regularization to apply.')
    parser.add_argument('--add-noise', help='The intensity of white noise to add to the signal (stdev)', default=0,
                        type=float)

    # parser.add_argument('--pooling', '-p', type=int, default=10)
    parser.add_argument('--dropout', '-do', type=float, default=0.4)
    parser.add_argument('--no-alignment', action='store_true', help='Whether to skip euclidean alignment.')
    parser.add_argument('--weight-decay', '-l2', type=float, default=0.01)
    parser.add_argument('--grad-clip', '-gc', type=float, default=0.)
    parser.add_argument('--model-param-dict', type=str, default=None)

    args = parser.parse_args()

    # Manual error parsings
    if args.fine_tune:
        if args.load_params is None or not (Path(args.load_params).is_dir() and Path(args.load_params).exists()):
            print("Param directory required for fine-tuning.")
            if not Path(args.load_params).is_dir():
                print("{} is not a directory".format(args.load_params))
            elif not Path(args.load_params).exists():
                print("{} does not exist".format(args.load_params))
            parser.error("Exiting.")
    # -----------------------------------------------------

    mne.set_log_level(False)
    try:
        args.func(args)
    except RuntimeError as e:
        print(e)
    except NaNError:
        print("Training ended due to NaN loss.")
