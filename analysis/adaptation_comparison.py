from utils import *
import numpy as np
import mne
import argparse

from datasets.mmi_physionet import load_eeg_bci
from datasets.ern import load_ern, ERN_STD_TEST_SUBJECTS

from train import reserve_training_from_test
from models import TIDNet


def generate_results(model: torch.nn.Module, test_subject, eval_subjects: OrderedDict, performance=None):
    def eval(test_ds):
        model.eval()

        preds = []
        true = []
        for vals in tqdm.tqdm(test_ds, unit='batch', desc='Evaluating'):
            val_x, val_y = vals[0].to('cuda'), vals[1].to('cuda').squeeze().long()
            model_out = model(val_x.unsqueeze(0))['prediction']
            preds.append(model_out.detach().cpu())
            true.append(val_y.cpu())
        return torch.cat(preds).squeeze(), torch.stack(true).squeeze()

    for subject in tqdm.tqdm(eval_subjects):
        if subject == test_subject:
            _, test = reserve_training_from_test(eval_subjects[test_subject])
            y_p, y_t = eval(test)
        else:
            y_p, y_t = eval(eval_subjects[subject])
        performance[test_subject][subject] = tensor_acc(y_t, y_p)

    return performance


def _mmi_performance(targets, param_dir, normalizer=fixed_scale, tmin=0, tlen=3, alignment=True, subjects=1,
                     runs=3):
    loaded_subjects = load_eeg_bci(targets, tmin, tlen, normalizer=normalizer, alignment=alignment)
    subject_folds = [list(x) for x in np.array_split(list(loaded_subjects.keys()), 5)]

    param_files = [param_dir + 'DSCNN_params_{}.pt'.format(i) for i in range(len(loaded_subjects))]
    model = TIDNet(targets=targets, runs=runs, subjects=subjects, samples=tlen * 160, channels=64).cuda()

    performance = pd.DataFrame(index=list(loaded_subjects.keys()), columns=list(loaded_subjects.keys()))

    for fold in tqdm.tqdm(subject_folds, desc='Folds'):
        for target_subject in tqdm.tqdm(fold, desc='Target Subject'):
            state_dict = torch.load(str(param_files.pop(0)))
            # Seen some problems here - comment next two lines if state dict works properly
            if 'classify.1.weight' in model.state_dict().keys() and 'classify.0.weight' in state_dict.keys():
                state_dict['classify.1.weight'] = state_dict.pop('classify.0.weight')
                state_dict['classify.1.bias'] = state_dict.pop('classify.0.bias')
            model.load_state_dict(state_dict, strict=False)
            performance = generate_results(model, target_subject, OrderedDict({s: loaded_subjects[s] for s in fold}),
                                           performance)

    return performance


def _ern_performance(param_dir, normalizer='fixedscale', tmin=-0.5, tlen=2, alignment=True, filter=False, subjects=1,
                     run=5):
    loaded_subjects = load_ern(tmin, tlen, normalizer=normalizer, alignment=alignment, filter=filter)
    subject_folds = [ERN_STD_TEST_SUBJECTS]

    param_files = [param_dir + 'DSCNN_params_{}.pt'.format(i) for i in range(len(ERN_STD_TEST_SUBJECTS))]
    model = TIDNet(targets=2, runs=run, subjects=subjects, samples=tlen * 200, channels=56).cuda()

    performance = pd.DataFrame(index=ERN_STD_TEST_SUBJECTS, columns=ERN_STD_TEST_SUBJECTS)

    for fold in tqdm.tqdm(subject_folds, desc='Folds'):
        for target_subject in tqdm.tqdm(fold, desc='Target Subject'):
            state_dict = torch.load(str(param_files.pop(0)))
            if 'classify.1.weight' in model.state_dict().keys() and 'classify.0.weight' in state_dict.keys():
                state_dict['classify.1.weight'] = state_dict.pop('classify.0.weight')
                state_dict['classify.1.bias'] = state_dict.pop('classify.0.bias')
            model.load_state_dict(state_dict, strict=False)
            performance = generate_results(model, target_subject, OrderedDict({s: loaded_subjects[s] for s in fold}),
                                           performance)

    return performance


def generate_mmi(seconds, targets):
    mne.set_log_level(False)

    for target in targets:
        perf = _mmi_performance(target, 'saved_models/MMI/{}s/fine_tuned-{}/'.format(seconds, target), alignment=True,
                                tlen=seconds)
        perf.to_csv('analysis/mmi_{}s_{}_ft.csv'.format(seconds, target))
        perf = _mmi_performance(target, 'saved_models/MMI/{}s/fine_tuned-{}/'.format(seconds, target), alignment=False,
                                tlen=seconds)
        perf.to_csv('analysis/mmi_{}s_{}_ft_no_ea.csv'.format(seconds, target))

        perf = _mmi_performance(target, 'saved_models/MMI/{}s/targetted_mdl_{}/'.format(seconds, target), tlen=seconds,
                                alignment=True, subjects=64)
        perf.to_csv('analysis/mmi_{}s_{}_mdl.csv'.format(seconds, target))
        perf = _mmi_performance(target, 'saved_models/MMI/{}s/targetted_mdl_{}/'.format(seconds, target), tlen=seconds,
                                alignment=False, subjects=64)
        perf.to_csv('analysis/mmi_{}s_{}_mdl_no_ea.csv'.format(seconds, target))


def generate_ern():
    perf_ern = _ern_performance('saved_models/ERN/ERN/fine-tuned/', alignment=True)
    perf_ern.to_csv('analysis/ern_ft.csv')
    perf_ern = _ern_performance('saved_models/ERN/ERN/targetted_mdl/', alignment=True, subjects=13, run=5)
    perf_ern.to_csv('analysis/ern_mdl.csv')
    # perf_ern = _ern_performance('saved_models/ERN/ERN/targetted_mdl/', alignment=False, subjects=13, run=5)
    # perf_ern.to_csv('analysis/ern_mdl_no_ea.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ern', action='store_true')
    parser.add_argument('--mmi', action='store_true')
    parser.add_argument('--mmi-targets', type=int, nargs='+', default=[2, 3, 4], choices=[2, 3, 4])
    parser.add_argument('--mmi-seconds', type=int, default=3, choices=[3, 6])
    args = parser.parse_args()

    if args.mmi:
        generate_mmi(args.mmi_seconds, args.mmi_targets)
    if args.ern:
        generate_ern()
