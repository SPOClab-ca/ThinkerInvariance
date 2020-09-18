import mne
import bisect
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict, Counter
import tqdm
from itertools import chain
from sklearn import metrics

from pathlib import Path
from mne.filter import filter_data, resample

import torch
from torch.nn.functional import log_softmax
from torch.nn.modules.loss import _Loss
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from xlrd import XLRDError

from ewma import ewma_vectorized_2d


class NaNError(BaseException):
    pass


class ModelQueue(list):
    def __init__(self, size):
        self._max_num = size
        super().__init__()

    def push(self, model_dict: OrderedDict):
        if len(self) == self._max_num:
            self.pop(-1)
        self.insert(0, model_dict)

    def ewma(self, weighting=0.5):
        weights = np.power(weighting, list(reversed(range(len(self)))))
        return self.smooth(weights)

    def average(self):
        weights = np.ones(len(self))
        return self.smooth(weights)

    def smooth(self, weights):
        weights = weights / weights.sum()
        new_model = OrderedDict()
        for p in self[0]:
            new_model[p] = torch.stack([weights[i] * self[i][p] for i in range(len(self))], dim=0).sum(dim=0)
        return new_model


def rand_split(dataset, frac=0.75):
    samples = len(dataset)
    return random_split(dataset, lengths=[round(x) for x in [samples*frac, samples*(1-frac)]])


def balanced_split(dataset, labels, frac=0.25):
    all_inds = np.arange(len(dataset))
    counts = np.round(np.bincount(labels) * frac)
    frac_inds = list()
    for i in all_inds:
        if counts[labels[i]] > 0:
            counts[labels[i]] -= 1
            frac_inds.append(i)
    frac_inds = np.array(frac_inds)
    remaining_inds = np.setdiff1d(all_inds, frac_inds)
    return Subset(dataset, frac_inds), Subset(dataset, remaining_inds)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor as numpy array"""
    return np.eye(num_classes, dtype='uint8')[y]


def make_results_df(method, num_train, targets, acc, run, **kwargs):
    required = dict(method=[method], num_train=num_train, targets=targets, acc=acc, run=run)
    required.update(kwargs)
    df = pd.DataFrame(required)
    df = df.set_index(list(required.keys()))
    return df


def update_pd(df1, df2):
    if df1 is None:
        return df2
    try:
        df1 = df1.append(df2, verify_integrity=True)
    except ValueError:
        df1.update(df2)
    return df1


def update_excel(file, df: pd.DataFrame, sheet):
    try:
        sheets = pd.read_excel(file, sheetname=None)
    except (FileNotFoundError, XLRDError):
        df.to_excel(file, sheet)
        return 
    with pd.ExcelWriter(file) as writer:
        for s in sheets:
            if s != sheet:
                sheets[s].to_excel(writer, s)
        df.to_excel(writer, sheet)


def train_val_test_subjectwise(all, test, num_validating=1):
    training = all.copy()

    testing = training[test]
    validating = list(range(len(training)))
    validating.remove(test)
    validating = [training[i] for i in np.random.choice(validating, num_validating, replace=False)]
    training.remove(testing)
    for s in validating:
        training.remove(s)

    return training, validating, testing


def one_hot(y: torch.Tensor, num_classes):
    """ 1-hot encodes a tensor to another similarly stored tensor"""
    if len(y.shape) > 0 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    out = torch.zeros(y.size()+torch.Size([num_classes]), device=y.device)
    return out.scatter_(-1, y.view((*y.size(), 1)), 1)


def auc_roc(y_t, y_p):
    fpr, tpr, thresholds = metrics.roc_curve(y_t, torch.exp(y_p[:, -1]), pos_label=1)
    return metrics.auc(fpr, tpr)


def zscore(data: np.ndarray, axis=-1):
    return (data - data.mean(axis, keepdims=True)) / (data.std(axis, keepdims=True) + 1e-12)


def fixed_scale(data: np.ndarray, axis=-1):
    return data / np.max(np.abs(data))


def ewma(data: np.ndarray, alpha=0.999, axis=-1):
    return exp_moving_whiten(data, factor_new=1-alpha)


NORMALIZERS = dict(zscore=zscore, ewma=ewma, fixedscale=fixed_scale, none=lambda x: x)


def exp_moving_whiten(data: np.ndarray, factor_new=0.001, init_block_size=1000, eps=1e-4):
    """
    This is very inefficent, but need this to properly reproduce previous work.

    Most code in this function taken from:
    https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signalproc.py
    :param data:
    :param factor_new:
    :return:
    """
    if data.shape[0] == 1:
        data = data.squeeze(0)
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return np.expand_dims(standardized.T, axis=0)


def tensor_acc(y_t, y_p: torch.Tensor):
    if len(y_p.shape) == 1:
        return torch.sum((y_p > 0).float() == y_t).item() / np.prod(y_t.size())
    return torch.sum(y_p.argmax(1) == y_t).item() / np.prod(y_t.size())


def print_stats(epoch, loss, acc, phase='Validation', print_fn=tqdm.tqdm.write, trend_factor=0.6):
    if trend_factor:
        # if not print_stats.loss_trend:
        #     print_stats.loss_trend = loss
        #     print_stats.acc_trend = acc
        print_stats.loss_trend = \
            trend_factor * loss + (1 - trend_factor) * print_stats.loss_trend if print_stats.loss_trend else loss
        print_stats.acc_trend = \
            trend_factor * acc + (1 - trend_factor) * print_stats.acc_trend if print_stats.acc_trend else acc

        print_fn('Epoch {} {} -- Loss: {:.4f} => {:.3f}, Accuracy: {:.2f}% => {:.2f}%'.format(
            epoch + 1, phase, loss, print_stats.loss_trend, 100 * acc, 100 * print_stats.acc_trend)
        )
    else:
        print_fn('Epoch {} Validation -- Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, loss, 100 * acc))

    return print_stats.loss_trend, print_stats.acc_trend


print_stats.loss_trend = None
print_stats.acc_trend = None


def remove_correlated_eog(raw: mne.io.Raw, eog_chs, picks=None):
    # ICA not stable with drifting signals
    tqdm.tqdm.write("Removing ICA components correlated to channels {}".format(eog_chs))
    if 'highpass' not in raw.info.keys() or raw.info['highpass'] < 1.0:
        tqdm.tqdm.write("High pass prefilter...")
        raw.filter(1.0, None, n_jobs=4, picks=picks)

    filename = Path(raw.filenames[0])
    ica_filename = filename.parent / (filename.stem + "-ica.fif.gz")
    if ica_filename.exists():
        tqdm.tqdm.write("Found existing ICA at {}".format(str(ica_filename)))
        ica = mne.preprocessing.read_ica(str(ica_filename))
    else:
        # Calculate components
        tqdm.tqdm.write("Fitting ICA...")
        ica = mne.preprocessing.ICA(n_components=len(raw.ch_names) // 2, method='fastica')
        ica.fit(raw, picks=picks)

        # Preference for components identified across all eog channels
        excludes = list()
        for ch in eog_chs:
            # Try more liberal thresholds if no correlations found
            for thresh in [3.0, 2.5, 2.0]:
                eog_inds, scores = ica.find_bads_eog(raw, ch_name=ch, threshold=thresh)
                if len(eog_inds) > 0:
                    break
            # No more than 2 components subtracted per EOG channel, take highest 2 scores
            excludes.append(eog_inds[:2])

        common_components = set.intersection(*[set(s) for s in excludes])
        # Fall back to most frequent components if no common components
        if len(common_components) == 0:
            excludes = [comp for comp, cnt in Counter([idx for ex in excludes for idx in ex]).most_common()]
        else:
            excludes = list(common_components)

        # No more than 3 total components
        ica.exclude = excludes[:3]
        ica.save(ica_filename)

    tqdm.tqdm.write("Dropping {} components".format(len(ica.exclude)))
    raw = ica.apply(raw)
    return raw


class SDLoss(_Loss):
    def __init__(self, features, decay_rate=0.9, size_average=True, reduce=True):
        super(SDLoss, self).__init__(size_average, reduce)
        self.register_buffer('c_accu', torch.zeros((1, features, features)))
        self.register_buffer('eye', torch.eye(features))
        self.decay_rate = decay_rate
        self.features = features

    def forward(self, z):
        bs = z.shape[0]
        m = z.shape[1]
        assert m == self.features
        c_mini = (1 / (m - 1)) * z.view(bs, m, -1) @ z.view(bs, m, -1).permute((0, 2, 1))
        self.c_accu = self.decay_rate * self.c_accu.detach() + (1 - self.decay_rate) * c_mini
        return torch.mean(torch.abs((c_mini * (1 - self.eye))))


class RampDecayLR(_LRScheduler):

    def __init__(self, optimizer, max_lr, warmup, decay_factor=0.5, last_epoch=-1):
        self.decay = -1.0 * decay_factor
        self.scale = max_lr / np.power(warmup, self.decay)
        self.warmup = warmup
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [self.scale * min(np.power(self.last_epoch + 2, self.decay),
                                 (self.last_epoch + 2) * np.power(self.warmup, self.decay - 1))
                for _ in self.base_lrs]


class LinearUpDownLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, warmup, total_epochs=100, last_epoch=-1):
        self.step_size = max_lr / (total_epochs - warmup)
        self.max_lr = max_lr
        self.warmup = warmup
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [min(self.max_lr * (self.last_epoch + 1) / self.warmup,
                    -1 * self.step_size * (self.last_epoch - self.warmup + 1) + self.max_lr)
                for _ in self.base_lrs]


class CommonDecay(_LRScheduler):
    def __init__(self, optimizer, max_lr, decay=0.5, rate=1, last_epoch=-1):
        self.max_lr = max_lr
        self.decay = decay
        self.rate = rate
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [self.max_lr * np.power(self.decay, self.last_epoch // self.rate) for _ in self.base_lrs]


class CosineDecay(_LRScheduler):
    def __init__(self, optimizer, max_lr, warmup, total_epochs=100, warm_drop=1.0, last_epoch=-1):
        self.real_eps = total_epochs - warmup
        self.max_lr = max_lr
        self.warmup = warmup
        self.warm_drop = warm_drop
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        progress = self.last_epoch - self.warmup + 1
        if progress <= 0:
            return [self.max_lr * (self.last_epoch + 1) / self.warmup for _ in self.base_lrs]
        else:
            return [self.warm_drop * self.max_lr * 0.5 * (1 + np.cos(progress * np.pi / self.real_eps))
                    for _ in self.base_lrs]


def normal_pdf_d(mu, stdev, num):
    x = np.zeros(num)
    denom = 1 / (stdev * np.sqrt(2*np.pi))
    var = np.power(stdev, 2)
    for i in range(num):
        x[i] = denom * np.exp(-1 * np.power(i - mu, 2) / (2 * var))
        x[mu] += 1.0 - x.sum()
    return x.astype('float32')


class SamplesDataset(Dataset):

    def __init__(self, dataset, stdev=10):
        self.dataset = dataset
        self.stdev = stdev
        self.center = len(self) // 2
        self.dist = torch.tensor(normal_pdf_d(self.center, stdev, len(self)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        s = self.dist.roll(index - self.center)
        return x, y, s


class MultiInputDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        x, y = [], []
        for ds in self.datasets:
            _x, _y = ds[index]
            x.append(_x)
            y.append(_y)
        assert all(i == y[0] for i in y)
        return (*x, y[0])

    def __len__(self):
        return min(len(x) for x in self.datasets)


class CrossValidator:

    def __init__(self, dataset: Dataset, folds=5):
        self.dataset = dataset
        self.folds = folds
        self._current_fold = 0
        self.indexes = np.split(np.arange(len(dataset)), folds)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_fold >= self.folds:
            raise StopIteration
        else:
            self._current_fold += 1
            val_inds = self.indexes[self._current_fold - 1]
            train_inds = np.concatenate([x for i, x in enumerate(self.indexes) if i != self._current_fold - 1]).tolist()
            return Subset(self.dataset, train_inds), Subset(self.dataset, val_inds)

    def __len__(self):
        return self.folds

    @property
    def fold(self):
        return self._current_fold - 1


class DemeanTransform(object):
    """
        """

    def __init__(self, scale, axis=1):
        self.scale = scale
        self.axis = axis

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (B, C, H, *W) to be normalized.

        Returns:
            Tensor: Demeaned and scaled tensor
        """
        return (tensor - tensor.mean(self.axis)) / self.scale

    def __repr__(self):
        return self.__class__.__name__ + '(scale={0}, axis={1})'.format(self.scale, self.axis)


class ExtendToTransform(object):
    """
    Transform to extend a tensor to the desired shape.
    The methods to accomplish this:
       - *tile* the datapoint so that it fits the desired shape (method='tile')
       - *pad* with a constant value (method=<float to right-pad>)
       - *interpolate* TODO

    """
    _METHODS = ['repeat']

    def __init__(self, max_size, method='repeat'):
        if isinstance(max_size, int):
            max_size = [max_size]
        self.max_size = max_size
        if isinstance(method, (float, int)) or method in self._METHODS:
            self.method = method
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (C, H, *W) to be normalized.

        Returns:
            Tensor: Demeaned and scaled tensor
        """
        curr_shape = tensor.shape[-len(self.max_size):]
        scale = np.ceil(np.divide(self.max_size, curr_shape)).astype(int)
        inds = [slice(0, x) for x in [*tensor.shape[:-len(self.max_size)], *self.max_size]]
        return np.tile(tensor, scale)[inds]

    def __repr__(self):
        return self.__class__.__name__ + '(max_size={0}, method={1})'.format(self.max_size, self.method)


class BandpassResampleTransform(object):
    """
        Bandpass and resample to the Nyquist frequency
        The methods to accomplish this:
           - *tile* the datapoint so that it fits the desired shape (method='tile')
           - *pad* with a constant value (method=<float to right-pad>)
           - *interpolate* TODO

        """
    _METHODS = ['repeat']

    def __init__(self, sfreq, high, low=None, njobs=10, nyq_factor=1.0):
        self.sfreq = sfreq
        self.high = high
        self.low = low
        self.njobs = njobs
        self.downsample = sfreq / (nyq_factor * 2 * self.high)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (..., T) to be filtered and downsampled.

        Returns:
            Tensor: Filtered and downsampled channels
        """
        tensor = filter_data(tensor.astype(np.float64), self.sfreq, l_freq=self.low, h_freq=self.high,
                             n_jobs=self.njobs, verbose=False)
        return resample(tensor, down=self.downsample, verbose=False).astype(np.float32)

    def __repr__(self):
        return self.__class__.__name__ + '(max_size={0}, method={1})'.format(self.max_size, self.method)


class TemporalCroppingTransform(object):
    """

    """

    def __init__(self, crop_len, window=(0, -1), test_offset=-1):
        self.crop_len = round(crop_len)
        self.window = [window[0], window[1]-crop_len]
        self.test_offset = test_offset

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (..., T) to be cropped to T' (T' <= T).

        Returns:
            Tensor: Cropped Tensor
        """
        if self.test_offset >= 0:
            offset = self.test_offset
        else:
            window = self.window if self.window[1] > 0 else [self.window[0], tensor.shape[-1]+self.window[1]]
            offset = np.random.choice(np.arange(*window)).astype(int)
        assert offset+self.crop_len < tensor.shape[-1]
        return tensor[..., offset:offset+self.crop_len]

    def __repr__(self):
        return self.__class__.__name__ + '(max_size={0}, method={1})'.format(self.max_size, self.method)
