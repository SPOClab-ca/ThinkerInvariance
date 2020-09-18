from utils import *
from dataload import TOPLEVEL, EpochsDataset, EuclideanAlignment, MultiDomainDataset, BasePreprocess
from train import full_training, loso_split, balanced_undersampling, subject_specific

import torch
import parse
import mne
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, balanced_accuracy_score

TOPLEVEL_SPELL = TOPLEVEL / 'erpbci' / '1.0.0'
SUBJECTS_SPELLER = ['s{0:02d}'.format(i) for i in range(1, 13)]

# Added the '#' character due to its inexplicable presence in the annotations
SPELLER_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F',
                   'G', 'H', 'I', 'J', 'K', 'L',
                   'M', 'N', 'O', 'P', 'Q', 'R',
                   'S', 'T', 'U', 'V', 'W', 'X',
                   'Y', 'Z', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9', '_', '#']
_CONVENIENT_EYE = np.eye(len(SPELLER_OPTIONS)).astype(int)

# 120 flashes at 150ms SOA
MAX_ACCEPTABLE_FLASHES = 144
SOA = 0.15
TOTAL_RUN_TIME_S = int(MAX_ACCEPTABLE_FLASHES * SOA)
MAX_RUNS = 21


TEST_METRICS = dict(
    AUROC=auc_roc
)


class BioSemi64SimpleMap(BasePreprocess):
    _MAP = [
               -1, -1, -1, -1, 'Fp1', 'Fpz', 'Fp2', -1, -1, -1, -1,
               -1, -1, -1, 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', -1, -1, -1,
               -1, 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', -1,
               -1, 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', -1,
               -1, 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', -1,
               -1, 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', -1,
               'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
               -1, -1, -1, 'PO7', 'PO3', 'POz', 'PO4', 'PO8', -1, -1, -1,
               -1, -1, -1, -1, 'O1', 'Oz', 'O2', -1, -1, -1, -1,
               -1, -1, -1, -1, -1, 'Iz', -1, -1, -1, -1, -1
    ]

    def __init__(self, epochs: mne.Epochs):
        super().__init__(epochs)
        self._map = self._MAP.copy()
        for i in range(len(self._map)):
            if self._map[i] != -1:
                self._map[i] = epochs.ch_names.index(self._map[i])

    def __call__(self, x: np.ndarray):
        x = np.concatenate([x, np.zeros((1, 1) + x.shape[2:])], axis=1)
        return x[:, self._map, ...].reshape(1, 10, 11, -1)


def load_subjects(normalizer: lambda x: x):
    raws = list()
    for subject in tqdm.tqdm(SUBJECTS_SPELLER, desc='Loading Data'):
        subject_list = list()
        for run in (TOPLEVEL_SPELL / subject).glob('*.raw.fif'):
            raw = mne.io.read_raw_fif(str(run), preload=True)
            # raw = raw.apply_function(normalizer, channel_wise=False)
            # raw = mne.io.read_raw_edf(str(run), preload=True)
            # raw = raw.filter(None, 120)
            # f = str(run.with_suffix('.raw.fif'))
            # raw.save(f, overwrite=True)
            subject_list.append(raw)
        raws.append(subject_list)
    return raws


def _get_target_and_crop(raw: mne.io.Raw):
    target_char = parse.search('#Tgt{}_', raw.annotations[0]['description'])[0]

    # Find the first speller flash (it isn't consistently at the second or nth index for that matter)
    start_off = 0
    while len(raw.annotations[start_off]['description']) > 6 and start_off < len(raw.annotations):
        start_off += 1
    assert start_off < len(raw.annotations) - 1
    start_t = raw.annotations[start_off]['onset']
    end_t = start_t + TOTAL_RUN_TIME_S
    # Operates in-place
    raw.crop(start_t, end_t, include_tmax=False)
    return target_char


def _one_hot_letter(letter):
    return _CONVENIENT_EYE[:, SPELLER_OPTIONS.index(letter.upper())]


def single_trials(loaded, tmin=0, tlen=1.0, decimate=32, normalizer=zscore, map_2d=False, ica_reject=False, eog=False,
                  ref=True):
    loadable = OrderedDict()
    for i, subject in tqdm.tqdm(enumerate(loaded), desc='Preprocessing Subjects', total=len(loaded)):
        all_runs = list()
        if len(subject[0].annotations) == 0:
            tqdm.tqdm.write('No annotations for subject {} '.format(SUBJECTS_SPELLER[i]))
            continue
        for run in tqdm.tqdm(subject, desc='Epoch and Filtering Runs'):
            target_letter = _get_target_and_crop(run)
            picks = list(range(64))
            if ref:
                picks += [64, 65]
            if not eog:
                picks += [66, 67, 68, 69]
            if ica_reject:
                run = remove_correlated_eog(run, run.ch_names[66:], picks)
            events, occurrences = mne.events_from_annotations(run, lambda a: int(target_letter in a))
            all_runs.append(mne.Epochs(run, events, tmin=tmin, tmax=tmin + tlen - 1 / run.info['sfreq'], baseline=None,
                                       decim=decimate, reject_by_annotation=False, picks=picks))
        loadable[SUBJECTS_SPELLER[i]] = MultiDomainDataset(
            [EpochsDataset(e, preproccesors=[EuclideanAlignment, BioSemi64SimpleMap] if map_2d else EuclideanAlignment,
                           normalizer=normalizer) for e in all_runs], force_num_domains=MAX_RUNS)
    return loadable


def p300_speller_system(args):
    all_loaded = load_subjects(normalizer=zscore)

    args.tmin = args.tmin if args.tmin is not None else -0.05
    args.tlen = args.tlen if args.tlen is not None else 0.7
    args.teval = args.teval if args.teval is not None else args.tmin

    args.separate_runs = max([len(x) for x in all_loaded])
    all_loaded = single_trials(all_loaded, args.tmin, args.tlen, decimate=args.decimate, eog=args.include_eog,
                               ica_reject=args.ica_eog, normalizer=NORMALIZERS[args.normalizer], map_2d=args.map_2d,
                               ref=not args.drop_ref)
    args.targets = 2
    ex = list(all_loaded.values())[0].datasets[0]
    args.channels, args.sfreq = ex.channels, ex.sfreq
    full_training(args, all_loaded, "P300", split_func=subject_specific if args.subject_specific else loso_split,
                  train_sampler=balanced_undersampling, test_metrics=TEST_METRICS, evaluation_batch_scaler=1)



