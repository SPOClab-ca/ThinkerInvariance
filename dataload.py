import mne
from typing import Iterable
from scipy.linalg import sqrtm, inv

from pathlib import Path
from utils import *

TOPLEVEL = Path('/ais/clspace5/spoclab/BrainData/')


class BasePreprocess(object):

    def __init__(self, epochs: mne.Epochs):
        self.epochs = epochs

    def __call__(self, x: np.ndarray):
        raise NotImplementedError()


class ScalePreprocess(BasePreprocess):

    def __init__(self, epochs, scale=1000):
        super().__init__(epochs)
        self.scale = scale

    def __call__(self, x):
        return self.scale * x


class EuclideanAlignment(BasePreprocess):

    def __init__(self, epochs: mne.Epochs = None, data=None):
        if data is not None:
            x = data
        elif epochs is not None:
            x = epochs.get_data()
        else:
            raise ValueError('Either epochs or data must be specified.')
        r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
        self.r_op = inv(sqrtm(r))
        if np.iscomplexobj(self.r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            self.r_op = np.real(self.r_op).astype(np.float32)
        elif not np.any(np.isfinite(self.r_op)):
            print("WARNING! Not finite values in R Matrix")

    def __call__(self, x: np.ndarray):
        return np.matmul(self.r_op, x)


def standard_epochsdataset(raw: mne.io.Raw, tmin=0, tlen=3, trial_normalizer=zscore, baseline=None, annot_resolver=None,
                           euclidean_alignment=True, runs=None, subject_normalizer=None, filter_bp=None, decim=1,
                           pick_eog=False, reject_eog_by_ica=None):
    picks = mne.pick_types(raw.info, stim=False, meg=False, eeg=True, emg=False, fnirs=False, eog=pick_eog)
    sfreq = raw.info['sfreq']
    if subject_normalizer:
        raw.load_data()
        raw = raw.apply_function(subject_normalizer, channel_wise=False)
    if filter_bp is not None:
        if isinstance(filter_bp, (list, tuple)) and len(filter_bp) == 2:
            raw.load_data()
            raw.filter(filter_bp[0], filter_bp[1], picks=picks, n_jobs=4)
        else:
            print('Filter must be provided as a two-element list [low, high]')

    if reject_eog_by_ica is not None:
        raw = remove_correlated_eog(raw, reject_eog_by_ica, picks=picks)

    events = mne.events_from_annotations(raw, event_id=annot_resolver)[0] if annot_resolver is not None \
        else mne.find_events(raw)

    # Map various event codes to a incremental label system
    _, events[:, -1] = np.unique(events[:, -1], return_inverse=True)

    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / sfreq, preload=True, decim=decim,
                        picks=picks, baseline=None, reject_by_annotation=False)
    return EpochsDataset(epochs, preproccesors=EuclideanAlignment if euclidean_alignment else [],
                         normalizer=trial_normalizer, runs=runs)


def epoch_and_tensorify(raws: dict, tmin=0, tlen=3., trial_normalizer=zscore, baseline=None, annot_resolver=None,
                        decim=1, euclidean_alignment=True, runs=None, subject_normalizer=None, filter_bp=None,
                        pick_eog=False, reject_eog_by_ica=None):
    """
    Take a dictionary of MNE Raws, probably produced by the load_*_raw functions, epoch them according to tmin and len,
    then convert the data and targets into a TensorDataset.
    :param raws: Dictionary of MNE Raws
    :param tmin: Starting time (in seconds) with respect to stimulation channel events
    :param tlen: Length of epochs (in seconds)
    :param trial_normalizer: Function to apply to the epoched data, signature: f(data, axis=-1)
    :return: Updated dictionary with TensorDatasets rather than mne raws
    """
    for i in tqdm.tqdm(raws.keys(), desc="Epoch and Tensorify"):
        raws[i] = standard_epochsdataset(raws[i], tmin=tmin, tlen=tlen, trial_normalizer=trial_normalizer,
                                         baseline=baseline, annot_resolver=annot_resolver, decim=decim,
                                         euclidean_alignment=euclidean_alignment, runs=runs, filter_bp=filter_bp,
                                         subject_normalizer=subject_normalizer, pick_eog=pick_eog,
                                         reject_eog_by_ica=reject_eog_by_ica)
    return raws


class MultiDomainDataset(ConcatDataset):

    def __init__(self, datasets: Iterable[Dataset], force_num_domains=None):
        super().__init__(datasets)
        self.domains = len(datasets) if force_num_domains is None else force_num_domains

    def force_num_domains(self, num_domains):
        self.domains = num_domains

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return (*self.datasets[dataset_idx][sample_idx], one_hot(torch.tensor(dataset_idx).long(), self.domains))


class EpochsDataset(Dataset):

    def __init__(self, epochs: mne.Epochs, force_label=None, picks=None, preproccesors=None, normalizer=zscore,
                 runs=None, train_mode=False):
        self.mode = train_mode
        self.epochs = epochs
        self._t_len = epochs.tmax - epochs.tmin
        self.loaded_x = [None for _ in range(len(epochs.events))]
        self.runs = runs
        self.picks = picks
        self.force_label = force_label if force_label is None else torch.tensor(force_label)
        self.normalizer = normalizer
        self.preprocessors = preproccesors if isinstance(preproccesors, (list, tuple)) else [preproccesors]
        for i, p in enumerate(self.preprocessors):
            self.preprocessors[i] = p(self.epochs)

    @property
    def channels(self):
        if self.picks is None:
            return len(self.epochs.ch_names)
        else:
            return len(self.picks)

    @property
    def sfreq(self):
        return self.epochs.info['sfreq']

    def train_mode(self, mode=False):
        self.mode = mode

    def __getitem__(self, index):
        ep = self.epochs[index]
        if self.loaded_x[index] is None:
            x = ep.get_data()
            if len(x.shape) != 3 or 0 in x.shape:
                print("I don't know why: {} index{}/{}".format(self.epochs, index, len(self)))
                print(self.epochs.info['description'])
                # raise AttributeError()
                return self.__getitem__(index - 1)
            x = x[0, self.picks, :]
            for p in self.preprocessors:
                x = p(x)
            x = torch.from_numpy(self.normalizer(x).astype('float32')).squeeze(0)
            self.loaded_x[index] = x
        else:
            x = self.loaded_x[index]

        y = torch.from_numpy(ep.events[..., -1]).long() if self.force_label is None else self.force_label

        if self.runs is not None:
            return x, y, one_hot(torch.tensor(self.runs * index / len(self)).long(), self.runs)

        return x, y

    def __len__(self):
        events = self.epochs.events[:, 0].tolist()
        return len(events)
