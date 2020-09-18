from utils import *
import dataload
from train import full_training, lmso_split, subject_specific, balanced_undersampling

import mne

TOPLEVEL_ERN = dataload.TOPLEVEL / 'ERN-inria/'
_FORMATTED_DIR = TOPLEVEL_ERN / 'mne_formatted'
SUBJECTS_ERN = ['S{0:02d}'.format(i) for i in range(1, 27)]
SESSIONS = 5

_TRUE_LABELS_URL = 'https://storage.googleapis.com/kaggle-forum-message-attachments/80787/2570/true_labels.csv'

ERN_STD_TEST_SUBJECTS = [SUBJECTS_ERN[i-1] for i in [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]]


TEST_METRICS = dict(
    AUCROC=auc_roc,
    BAC=lambda t, p: metrics.balanced_accuracy_score(t, p.argmax(1)),
)


def _pre_parse():
    # Used to initially parse the data provided by kaggle:
    #
    # 1. Before running, be sure to extract all csv (training and test) to TOPLEVEL_ERN.
    # 2. Make sure TrainLabels.csv is in the same directory
    # 3. In TOPLEVEL_ERN run the following to obtain the labels for the test subjects:
    #  wget https://storage.googleapis.com/kaggle-forum-message-attachments/80787/2570/true_labels.csv

    train_labels = np.loadtxt(TOPLEVEL_ERN / 'TrainLabels.csv', skiprows=1, usecols=1, delimiter=',')
    train_pt = 0
    test_labels = np.loadtxt(TOPLEVEL_ERN / 'true_labels.csv')
    test_pt = 0

    _FORMATTED_DIR.mkdir(exist_ok=True)
    for subject in tqdm.tqdm(SUBJECTS_ERN, desc='Preparsing CSV', unit='subject'):
        for sess in tqdm.trange(1, SESSIONS + 1):
            run = pd.read_csv(str(TOPLEVEL_ERN / 'Data_{}_Sess0{}.csv'.format(subject, sess)))
            chs = run.columns.tolist()[1:]
            ch_types = ['eeg'] * (len(chs) - 2) + ['eog', 'stim']
            arr = run[chs].values.T

            # Assign proper event labels very carefully
            event_mask = arr[-1, :] == 1
            if subject in ERN_STD_TEST_SUBJECTS:
                arr[-1, event_mask] += test_labels[test_pt:test_pt + int(event_mask.sum())]
                test_pt += int(event_mask.sum())
            else:
                arr[-1, event_mask] += train_labels[train_pt:train_pt + int(event_mask.sum())]
                train_pt += int(event_mask.sum())

            info = mne.create_info(chs, 200, ch_types)
            raw = mne.io.RawArray(arr, info)

            raw.save(str(_FORMATTED_DIR / '{}_{}.raw.fif'.format(subject, sess)))

    assert train_pt == len(train_labels) and test_pt == len(test_labels)


def competition_split(args, loaded_subjects: OrderedDict, **kwargs):
    non_test_subjects = [x for x in list(loaded_subjects.keys()) if x not in ERN_STD_TEST_SUBJECTS]
    remaining_folds = [list(x) for x in np.array_split(non_test_subjects, args.xval_folds)]

    # only one fold forces just the testing fold to be used, and full test iterates over permutations of the remaining
    subject_folds = [ERN_STD_TEST_SUBJECTS, *remaining_folds]
    args.xval_folds = 1
    args.full_test = True

    yield from lmso_split(args, loaded_subjects, subject_folds=subject_folds)


def load_ern(tmin=-0.5, tlen=2, normalizer='fixedscale', alignment=True, filter=False, ica_reject=False, eog=False):
    if not _FORMATTED_DIR.exists():
        _pre_parse()

    loaded = OrderedDict()

    for subject in tqdm.tqdm(SUBJECTS_ERN, desc='Loading Subjects', unit='subject'):
        runs = []
        for sess in range(1, SESSIONS + 1):
            raw = mne.io.read_raw_fif(str(_FORMATTED_DIR / '{}_{}.raw.fif'.format(subject, sess)), preload=True)
            runs.append(dataload.standard_epochsdataset(raw, tmin, tlen, trial_normalizer=NORMALIZERS[normalizer],
                                                        euclidean_alignment=alignment, pick_eog=eog,
                                                        filter_bp=[1, 40] if filter else None,
                                                        decim=1, reject_eog_by_ica='EOG' if ica_reject else None))
        loaded[subject] = dataload.MultiDomainDataset(runs)
    return loaded


def ern(args):
    # Default Epoch parameters
    args.tmin = args.tmin if args.tmin is not None else -0.5
    args.tlen = args.tlen if args.tlen is not None else 2
    args.teval = args.teval if args.teval is not None else args.tmin

    loaded_subjects = load_ern(args.tmin, args.tlen, args.normalizer, not args.no_alignment, args.filter,
                               eog=args.include_eog, ica_reject=args.ica_eog)

    args.channels, args.sfreq = 56 + int(args.include_eog), 200
    args.targets = 2
    args.separate_runs = SESSIONS

    split_func = subject_specific if args.subject_specific else competition_split
    full_training(args, loaded_subjects, 'ERN', evaluation_batch_scaler=4, split_func=split_func,
                  train_sampler=balanced_undersampling, test_metrics=TEST_METRICS)
