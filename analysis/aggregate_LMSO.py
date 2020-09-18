import mne
from utils import *

from datasets.mmi_physionet import load_eeg_bci, SUBJECTS_EEGBCI
from datasets.ern import load_ern, ERN_STD_TEST_SUBJECTS

from train import reserve_training_from_test
from models import TIDNet, reEEGNet

CONFIGS = ['', '_mixup', '_ea', '_ea_mixup']


def generate_results(model: torch.nn.Module, test_subjects: OrderedDict, performance, perf_column, labels=None,
                     metric=tensor_acc):
    if labels is None:
        labels = list(test_subjects.keys())

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

    for i, subject in enumerate(tqdm.tqdm(test_subjects)):
        y_p, y_t = eval(test_subjects[subject])
        performance[perf_column][labels[i]] = metric(y_t, y_p)

    return performance


def load_and_evaluate(test_subject_sets, label_set, data_loader, param_dir, dest, targets, samples, channels, subjects,
                      runs, metric=tensor_acc):

    for name, model in tqdm.tqdm([('TIDNet', TIDNet), ('EEGNet', reEEGNet)]):
        model = model(targets=targets, runs=runs, subjects=subjects, samples=samples, channels=channels).cuda()
        performance = pd.DataFrame(index=[s for f in label_set for s in f], columns=CONFIGS)
        for config in tqdm.tqdm(CONFIGS):
            if config == '':
                loaded_subjects = data_loader(alignment=False)
            elif config == '_ea':
                loaded_subjects = data_loader(alignment=True)

            # This is some A+ style here
            param_files = [param_dir.format(name, config) + '{}_params_{}.pt'.format(
                'DSCNN' if name == 'TIDNet' else name, i)
                           for i in range(len(test_subject_sets))]

            for i, fold in enumerate(tqdm.tqdm(test_subject_sets, desc='Folds')):
                try:
                    model.load_state_dict(torch.load(str(param_files.pop(0))))
                except RuntimeError:
                    print("Wut")
                if 'mdl' in param_dir:
                    fold_subjects = OrderedDict({s: reserve_training_from_test(loaded_subjects[s])[1] for s in fold})
                else:
                    fold_subjects = OrderedDict({s: loaded_subjects[s] for s in fold})

                performance = generate_results(model, fold_subjects, performance, config, labels=label_set[i],
                                               metric=metric)
        performance.to_excel(dest.format(name))


if __name__ == '__main__':
    mne.set_log_level(False)

    print("ERN")
    ern_labels = [['_'.join([s, str(f)]) for s in ERN_STD_TEST_SUBJECTS] for f in range(4)]
    load_and_evaluate([ERN_STD_TEST_SUBJECTS] * 4, ern_labels, load_ern, 'saved_models/ERN/ERN/loso-{}{}/',
                      'analysis/aggregated/ERN_{}.xlsx', targets=2, channels=56, samples=2 * 200, subjects=12, runs=5,
                      metric=auc_roc)
    #
    print("ERN MDL")
    load_and_evaluate([ERN_STD_TEST_SUBJECTS] * 4, ern_labels, load_ern, 'saved_models/ERN/ERN/loso-{}{}_mdl/',
                      'analysis/aggregated/ERN_{}_mdl.xlsx', targets=2, channels=56, samples=2 * 200, subjects=21,
                      runs=5, metric=auc_roc)

    # print("MMI")
    # mmi_folds = [list(x) for x in np.array_split(SUBJECTS_EEGBCI, 5)]
    # print("MMI 2")
    # load_and_evaluate(mmi_folds, mmi_folds, lambda **kwargs: load_eeg_bci(targets=2, **kwargs),
    #                   'saved_models/MMI/3s/lmso-2-{}{}/', 'analysis/aggregated/MMI_2_{}.xlsx', targets=2, channels=64,
    #                   samples=3 * 160, subjects=63, runs=3)
    #
    # print("MMI 2 MDL")
    # load_and_evaluate(mmi_folds, mmi_folds, lambda **kwargs: load_eeg_bci(targets=2, **kwargs),
    #                   'saved_models/MMI/3s/lmso-2-{}{}_mdl/', 'analysis/aggregated/MMI_2_{}_mdl.xlsx', targets=2,
    #                   channels=64, samples=3 * 160, subjects=105, runs=3)
    #
    # print("MMI 3")
    # load_and_evaluate(mmi_folds, mmi_folds, lambda **kwargs: load_eeg_bci(targets=3, **kwargs),
    #                   'saved_models/MMI/3s/lmso-3-{}{}/', 'analysis/aggregated/MMI_3_{}.xlsx', targets=3, channels=64,
    #                   samples=3 * 160, subjects=63, runs=3)
    #
    # print("MMI 3 MDL")
    # load_and_evaluate(mmi_folds, mmi_folds, lambda **kwargs: load_eeg_bci(targets=3, **kwargs),
    #                   'saved_models/MMI/3s/lmso-3-{}{}_mdl/', 'analysis/aggregated/MMI_3_{}_mdl.xlsx', targets=3,
    #                   channels=64, samples=3 * 160, subjects=105, runs=3)
    #
    # print("MMI 4")
    # load_and_evaluate(mmi_folds, mmi_folds, lambda **kwargs: load_eeg_bci(targets=4, **kwargs),
    #                   'saved_models/MMI/3s/lmso-4-{}{}/', 'analysis/aggregated/MMI_4_{}.xlsx', targets=4, channels=64,
    #                   samples=3 * 160, subjects=63, runs=3)
    #
    # print("MMI 4 MDL")
    # load_and_evaluate(mmi_folds, mmi_folds, lambda **kwargs: load_eeg_bci(targets=4, **kwargs),
    #                   'saved_models/MMI/3s/lmso-4-{}{}_mdl/', 'analysis/aggregated/MMI_4_{}_mdl.xlsx', targets=4,
    #                   channels=64, samples=3 * 160, subjects=105, runs=3)

