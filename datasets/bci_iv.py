from utils import *
import dataload
from train import full_training, loso_split, subject_specific, balanced_undersampling

import mne

# Before using this dataset, we reformatted each session into the mne file format .raw.fif
TOPLEVEL_BCI_IV = dataload.TOPLEVEL / 'BCI-Competition-IV/'
SUBJECTS_BCI_IV = ['A{0:02d}'.format(i) for i in range(1, 10)]
EOG_CHS = ['EOG-left', 'EOG-central', 'EOG-right']


def load_bci_iv_2a_raw():
    toplevel_2a = TOPLEVEL_BCI_IV / '2a' / 'mne_ready'
    train = dict()
    test = dict()
    for s in tqdm.tqdm(SUBJECTS_BCI_IV, desc='Loading Subjects'):
        train[s] = mne.io.read_raw_fif(str(toplevel_2a / s) + 'T.raw.fif', preload=True)
        test[s] = mne.io.read_raw_fif(str(toplevel_2a / s) + 'E.raw.fif', preload=True)
    return train, test


def bci_competitions(args):
    # Default Epoch parameters
    args.tmin = args.tmin if args.tmin is not None else -0.5
    args.tlen = args.tlen if args.tlen is not None else 4.5
    args.teval = args.teval if args.teval is not None else -0.5

    loaded_train, loaded_test = load_bci_iv_2a_raw()
    loaded_train, loaded_test = dataload.epoch_and_tensorify(loaded_train, tmin=args.tmin, tlen=args.tlen,
                                                             trial_normalizer=NORMALIZERS[args.normalizer],
                                                             euclidean_alignment=not args.no_alignment,
                                                             filter_bp=[4, 40] if args.filter else None,
                                                             subject_normalizer=ewma if args.ewma_normalize else None,
                                                             pick_eog=args.include_eog,
                                                             reject_eog_by_ica=EOG_CHS if args.ica_eog else None),\
                                dataload.epoch_and_tensorify(loaded_test, tmin=args.tmin, tlen=args.tlen,
                                                             trial_normalizer=NORMALIZERS[args.normalizer],
                                                             euclidean_alignment=not args.no_alignment,
                                                             filter_bp=[4, 40] if args.filter else None,
                                                             subject_normalizer=ewma if args.ewma_normalize else None,
                                                             pick_eog=args.include_eog,
                                                             reject_eog_by_ica=EOG_CHS if args.ica_eog else None)
    ex_dataset = list(loaded_train.items())[0][1]
    ex_dataset.train_mode(True)
    args.channels, args.sfreq = ex_dataset.channels, ex_dataset.sfreq
    args.targets = 4
    args.separate_runs = 2

    loaded_subjects = OrderedDict()
    for s in sorted(loaded_train.keys()):
        loaded_subjects[s] = dataload.MultiDomainDataset([loaded_train[s], loaded_test[s]])

    split_func = subject_specific if args.subject_specific else loso_split
    full_training(args, loaded_subjects, 'BCI-IV-2a', evaluation_batch_scaler=1, force_test_run=[1],
                  split_func=split_func, train_sampler=balanced_undersampling)
