import pandas as pd

from analysis.aggregate_LMSO import CONFIGS

PERFORMANCES = dict(
    BCI_2a=[
        "results/eog_free/BCI_2a/loso{}{}",
        "acc",
        "",
        "-ShallowConvNet",
    ],
    P300=[
        "results/eog_free/p300-multi/loso{}{}",
        "AUROC",
        "",
        "-EEGNet",

    ],
)


def load_performance(file_loc, col='acc'):
    data = pd.read_excel(file_loc, header=1)
    data = data.sort_values('run')
    return data[col].values


if __name__ == '__main__':
    for dataset in PERFORMANCES:
        for style in ['', '_mdl']:
            for model in PERFORMANCES[dataset][2:]:
                aggregated = pd.DataFrame(columns=CONFIGS)
                for config in CONFIGS:
                    aggregated[config] = load_performance(PERFORMANCES[dataset][0].format(model, config+style+'.xlsx'),
                                                          col=PERFORMANCES[dataset][1])
                aggregated.to_excel('analysis/aggregated/' + dataset + '_' + model + style + '.xlsx')
