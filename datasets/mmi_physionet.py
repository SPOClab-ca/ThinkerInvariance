from utils import *
from dataload import TOPLEVEL, EpochsDataset, EuclideanAlignment, MultiDomainDataset
from train import full_training, lmso_split, balanced_undersampling, subject_specific, multi_split_helper

import mne
import argparse
from mne.datasets import eegbci


TOPLEVEL_EEGBCI = TOPLEVEL
BAD_SUBJECTS_EEGBCI = [87, 89, 91, 99]
SUBJECTS_EEGBCI = list(i for i in range(109) if i not in BAD_SUBJECTS_EEGBCI)
EVENTS_EEGBCI = dict(hands=2, feet=3)
BASELINE_EYES_OPEN = [1]
BASELINE_EYES_CLOSED = [2]

MOTOR_FISTS = (3, 7, 11)
IMAGERY_FISTS = (4, 8, 12)
MOTOR_FEET = (5, 9, 13)
IMAGERY_FEET_V_FISTS = (6, 10, 14)


def load_eeg_bci(targets=4, tmin=0, tlen=3, t_ev=0, t_sub=None, normalizer=zscore, low_f=None, high_f=None,
                 alignment=True):

    paths = [eegbci.load_data(s+1, IMAGERY_FISTS, path=str(TOPLEVEL_EEGBCI), update_path=False) for s in SUBJECTS_EEGBCI]
    raws = [mne.io.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in path])
            for path in tqdm.tqdm(paths, unit='subj', desc='Loading')]
    datasets = OrderedDict()
    for i, raw in tqdm.tqdm(list(zip(SUBJECTS_EEGBCI, raws)), desc='Preprocessing'):
        if raw.info['sfreq'] != 160:
            tqdm.tqdm.write('Skipping..., sampling frequency: {}'.format(raw.info['sfreq']))
            continue
        raw.rename_channels(lambda x: x.strip('.'))
        if low_f or high_f:
            raw.filter(low_f, high_f, fir_design='firwin', skip_by_annotation='edge')
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=0, T2=1))
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        epochs = mne.Epochs(raw, events[:41, ...], tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'], picks=picks,
                            baseline=None, reject_by_annotation=False)#.drop_bad()
        if targets > 2:
            paths = eegbci.load_data(i + 1, BASELINE_EYES_OPEN, path=str(TOPLEVEL_EEGBCI), update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
            raw.rename_channels(lambda x: x.strip('.'))
            if low_f or high_f:
                raw.filter(low_f, high_f, fir_design='firwin', skip_by_annotation='edge')
            events = np.zeros((events.shape[0] // 2, 3)).astype('int')
            events[:, -1] = 2
            events[:, 0] = np.linspace(0, raw.info['sfreq'] * (60 - 2 * tlen), num=events.shape[0]).astype(np.int)
            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            eyes_epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'], picks=picks,
                                     baseline=None, reject_by_annotation=False)#.drop_bad()
            epochs = mne.concatenate_epochs([eyes_epochs, epochs])
        if targets > 3:
            paths = eegbci.load_data(i+1, IMAGERY_FEET_V_FISTS, path=str(TOPLEVEL_EEGBCI), update_path=False)
            raw = mne.io.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
            raw.rename_channels(lambda x: x.strip('.'))
            if low_f or high_f:
                raw.filter(low_f, high_f, fir_design='firwin', skip_by_annotation='edge')
            events, _ = mne.events_from_annotations(raw, event_id=dict(T2=3))
            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            feet_epochs = mne.Epochs(raw, events[:20, ...], tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'],
                                     picks=picks, baseline=None, reject_by_annotation=False)#.drop_bad()
            epochs = mne.concatenate_epochs([epochs, feet_epochs])

        datasets[i] = EpochsDataset(epochs, preproccesors=EuclideanAlignment if alignment else [],
                                    normalizer=normalizer, runs=3)

    return datasets


def num_training_regression(args, loaded_subjects: OrderedDict, subject_split_func=lmso_split,
                            force_test_run=None):
    """
    Train on subsets of subjects to determine reliance of performance on the number of subjects present.
    Parameters
    ----------
    args
    loaded_subjects
    subject_split_func

    Returns
    -------

    """

    for training, validating, testing in subject_split_func(args, loaded_subjects, force_test_run=force_test_run):
        num_subjects_per = np.logspace(0, np.log10(len(training.datasets)), num=args.subj_subsets).round().astype(np.int)
        # Ensure no duplicate subsets
        for i in range(len(num_subjects_per) - 1):
            while num_subjects_per[i] >= num_subjects_per[i+1]:
                num_subjects_per[i+1] += 1
        tqdm.tqdm.write('Num subjects in subsets: ' + str(list(num_subjects_per)))
        # max_mixup = args.mixup
        for i, num_subjects in enumerate(tqdm.tqdm(num_subjects_per, desc='Subset of subjects')):
            tqdm.tqdm.write('Training with subset of {} subjects: {} Trials'.format(
                num_subjects, training.cumulative_sizes[num_subjects-1]))
            args.subjects_used = num_subjects
            # Need to attenuate mixup proportionate to points
            # args.mixup = max_mixup * (i + 1) / args.subj_subsets
            # tqdm.tqdm.write('Mixup attenuated to: ' + str(args.mixup))
            training.force_num_domains(num_subjects)
            _sub_train = Subset(training, np.arange(training.cumulative_sizes[num_subjects-1]))
            yield _sub_train, validating, testing


def mmidb(args):
    # Default Epoch parameters
    args.tmin = args.tmin if args.tmin is not None else 0
    args.tlen = args.tlen if args.tlen is not None else 3
    args.teval = args.teval if args.teval is not None else args.tmin

    all_loaded = load_eeg_bci(targets=args.targets, normalizer=NORMALIZERS[args.normalizer], tmin=args.tmin,
                              tlen=args.tlen, alignment=not args.no_alignment)

    args.channels, args.sfreq = all_loaded[0].channels, all_loaded[0].sfreq
    args.separate_runs = 3

    split_func = lmso_split
    if args.subj_subsets is not None:
        split_func = num_training_regression
    elif args.subject_specific:
        split_func = subject_specific

    full_training(args, all_loaded, "MMI-DB", split_func=split_func, train_sampler=balanced_undersampling,
                  evaluation_batch_scaler=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subject-wise test performances for the MMI Database.")
    parser.add_argument("M")
