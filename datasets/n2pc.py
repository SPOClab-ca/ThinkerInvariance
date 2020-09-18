from utils import *
from dataload import TOPLEVEL, epoch_and_tensorify
from train import full_training, balanced_oversampling, subject_specific, loso_split

import mne
from sklearn.metrics import balanced_accuracy_score

TOPLEVEL_N2PC = TOPLEVEL / 'ltrsvp/1.0.0/'
SUBJECTS_N2PC = ['rsvp_5Hz_{0:02d}a'.format(i) for i in [2, 3, 4, 6] + list(range(8, 15))]

TEST_METRICS = dict(
    # Kappa=cohen_kappa_score,
    BAC=lambda t, p: balanced_accuracy_score(t, p.argmax(1)),
)


def load_n2pc():
    raws = OrderedDict()
    for subject in tqdm.tqdm(SUBJECTS_N2PC, unit='Subject'):
        raws[subject] = mne.io.read_raw_edf(str(TOPLEVEL_N2PC / '5-Hz' / subject) + '.edf', preload=True)
    return raws


def n2pc(args):
    args.tmin = args.tmin if args.tmin is not None else -0.05
    args.tlen = args.tlen if args.tlen is not None else 0.7
    args.teval = args.teval if args.teval is not None else args.tmin

    loaded = load_n2pc()

    def event_from_annotation(annotation: str):
        if 'T=1' in annotation:
            return 1
        elif 'T=0' in annotation:
            return 0
        else:
            return None

    # We decimate the data since the LPF is already 28
    loaded = epoch_and_tensorify(loaded, tmin=args.tmin, tlen=args.tlen, trial_normalizer=NORMALIZERS[args.normalizer],
                                 annot_resolver=event_from_annotation, euclidean_alignment=not args.no_alignment,
                                 decim=4)

    args.channels, args.sfreq = list(loaded.items())[0][1].channels, list(loaded.items())[0][1].sfreq
    args.separate_runs = 1
    args.targets = 2

    full_training(args, loaded, 'N2PC', test_metrics=TEST_METRICS, evaluation_batch_scaler=4,
                  train_sampler=balanced_oversampling,
                  split_func=subject_specific if args.subject_specific else loso_split)


