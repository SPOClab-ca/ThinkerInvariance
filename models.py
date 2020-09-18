from layers import *

from math import ceil
from collections import OrderedDict
from braindecode.models.eegnet import EEGNetv4
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet


class SCN(Module):

    def __init__(self, targets=4, filters=40, channels=22, samples=1500, subjects=1, runs=None, **kwargs):
        super().__init__()
        self.base_model = ShallowFBCSPNet(channels, targets, samples, final_conv_length='auto').create_network()

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.base_model(x)


class EEGNet(Module):

    def __init__(self, targets=4, filters=40, channels=22, samples=1500, subjects=1, runs=None, **kwargs):
        super().__init__()
        self.base_model = EEGNetv4(channels, targets, input_time_length=samples).create_network()

    def forward(self, x):
        return self.base_model(x)


class ShallowConvNet(Module):

    def __init__(self, targets=2, channels=64, filters=40, t_f_len=25, samples=960, do=0.5, pooling=15, **kwargs):
        super().__init__()
        after_size = 40 * samples // pooling
        self.convs = Sequential(
            Conv2d(1, filters, (1, t_f_len)),
            Conv2d(filters, filters, (channels, 1)),
            BatchNorm2d(filters),
        )

        self.avg_pool = AvgPool2d((1, 75), (1, pooling))

        self.classify = Sequential(
            Dropout(do),
            Conv2d(filters, targets, (1, 11)),
            Flatten(),
            LogSoftmax()
        )

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        x = self.convs(x)

        x = x.pow(2)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))

        print(self.classify(x).shape)
        return self.classify(x)

    @property
    def num_features(self):
        return 80


class AltShallow(Module):

    def __init__(self, targets=2, channels=64, samples=960, do=0.0, pooling=15, subjects=None, runs=None, **kwargs):
        super().__init__()
        after_size = 40 * samples // pooling
        self.network = Sequential(
            Conv2d(1, 40, (1, 31), padding=(0, 15)),
            ReLU(),
            Conv2d(40, 40, (channels, 1)),
            ReLU(),
            AvgPool2d((1, 15)),
            Flatten(),
            Linear(after_size, 80),
            ReLU(),
        )
        self.classify = Sequential(
            Linear(80, targets),
            LogSoftmax()
        )

        # Not in the original
        self.subj_classify = Sequential(Linear(80, subjects), LogSoftmax()) if subjects is not None else None
        self.run_classify = Sequential(Linear(80, runs), LogSoftmax()) if runs is not None else None

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        f = self.network(x)
        results = dict(prediction=self.classify(f), features=f)
        if self.subj_classify is not None:
            results['subject'] = self.subj_classify(f)
        if self.run_classify is not None:
            results['run'] = self.run_classify(f)
        return results

    @property
    def num_features(self):
        return 80


class reEEGNet(Module):

    def __init__(self, targets=2, channels=64, samples=960, do=0.5, pooling=8, F1=8, D=2,
                 t_len=65, F2=16, **kwargs):
        super().__init__()

        self.init_conv = Sequential(
            Expand(1),
            Conv2d(1, F1, (1, t_len), padding=(0, t_len // 2), bias=False),
            BatchNorm2d(F1)
        )

        self.depth_conv = Sequential(
            Conv2d(F1, D * F1, (channels, 1), bias=False, groups=F1),
            BatchNorm2d(D * F1),
            ELU(),
            AvgPool2d((1, pooling // 2)),
            Dropout(0.25)
        )
        samples = samples // (pooling // 2)

        self.sep_conv = Sequential(
            # Separate into two convs, one that doesnt operate across filters, one isolated to filters
            Conv2d(D*F1, D*F1, (1, 17), bias=False, padding=(0, 8), groups=D*F1),
            Conv2d(D*F1, F2, (1, 1), bias=False),
            BatchNorm2d(F2),
            ELU(),
            AvgPool2d((1, pooling)),
            Dropout(0.25)
        )
        samples = samples // pooling

        self._num_features = F2 * samples

        self.classifier = Sequential(
            Flatten(),
            Linear(self._num_features, targets),
            LogSoftmax(dim=-1)
        )

    @property
    def num_features(self):
        return self._num_features

    def forward(self, x):
        x = self.init_conv(x)
        x = self.depth_conv(x)
        x = self.sep_conv(x)

        return dict(prediction=self.classifier(x))


class _tidnet_features(Module):

    def __init__(self, s_growth=24, t_filters=32, channels=22, samples=1500, do=0.4, pooling=20,
                 temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3, summary=-1):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.temp_len = ceil(temp_span * samples)

        self.temporal = Sequential(
            Expand(axis=1),
            TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len),
            MaxPool2d((1, pooling)),
            Dropout2d(do),
        )
        summary = samples // pooling if summary == -1 else summary

        self.spatial = DenseSpatialFilter(channels, s_growth, spat_layers, in_ch=t_filters, dropout_rate=do,
                                          bottleneck=bottleneck)
        self.extract_features = Sequential(
            AdaptiveAvgPool1d(int(summary)),
            Flatten()
        )

        self._num_features = (t_filters + s_growth * spat_layers) * summary

    @property
    def num_features(self):
        return self._num_features

    def forward(self, x, **kwargs):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)


class TIDNet(Module):

    def __init__(self, targets=4, s_growth=24, t_filters=32, channels=22, samples=1500, do=0.4, pooling=15, subjects=1,
                 temp_layers=2, spat_layers=2, runs=None, temp_span=0.05, bottleneck=3, summary=-1, **kwargs):
        super().__init__()
        self.classes = targets
        self.channels = channels
        self.subjects = subjects
        self.runs = runs
        self.samples = samples
        self.temp_len = ceil(temp_span * samples)

        self.dscnn = _tidnet_features(s_growth=s_growth, t_filters=t_filters, channels=channels, samples=samples,
                                      do=do, pooling=pooling, temp_layers=temp_layers, spat_layers=spat_layers,
                                      temp_span=temp_span, bottleneck=bottleneck, summary=summary, **kwargs)

        self._num_features = self.dscnn.num_features

        self.classify = self._create_classifier(self.num_features, targets)

        self.subject_prediction = self._create_classifier(self.num_features, subjects)
        self.run_prediction = self._create_classifier(self.num_features, runs)

    def _create_classifier(self, incoming, targets):
        classifier = Linear(incoming, targets)
        init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        return Sequential(Flatten(), classifier, LogSoftmax(dim=-1))

    def forward(self, x, **kwargs):

        x = self.dscnn(x)

        subject = self.subject_prediction(x) if self.subjects is not None else None
        run = self.run_prediction(x) if self.runs is not None else None

        return dict(prediction=self.classify(x), subject=subject, run=run, features=x.view(x.size(0), -1))

    @property
    def num_features(self):
        return self._num_features

    def restricted_param_loading(self, params: OrderedDict, freeze=False):
        removal = list()
        for param in params:
            if 'classify' in param or 'prediction' in param:
                removal.append(param)
        for p in removal:
            params.pop(p)
        self.load_state_dict(params, strict=False)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            print('All layers frozen')

            print('New last layer added, and all others frozen.')
            self.classify = Sequential(
                Linear(self.num_features, self.classes),
                LogSoftmax(dim=-1)
            )
            self.subject_prediction = Sequential(
                Linear(self.num_features, self.subjects),
                LogSoftmax()
            )
            self.run_prediction = Sequential(
                Linear(self.num_features, self.runs),
                LogSoftmax()
            )
            print('New classifiers added.')


MODELS = {
    'CNN-CSP': SCN,
    'Dose': AltShallow,
    'EEGNet': reEEGNet,
    'TIDNet': TIDNet,
}
