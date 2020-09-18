import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from scipy.stats import linregress
from pathlib import Path
from matplotlib.ticker import PercentFormatter, MultipleLocator

TOPLEVEL = Path('../results/MMI/subject_regression/')
NUM_REGRESSION_POINTS = 20
MIN_REGRESSION_SUBJECTS = 5
COLOURS = ['xkcd:azure', 'xkcd:lightblue', 'xkcd:purple', 'xkcd:pink']


def log_regression(data: pd.DataFrame):
    data = data[data['num_train'] >= MIN_REGRESSION_SUBJECTS]
    x = data['num_train'].values
    y = data['acc'].values

    return linregress(np.log(x), y)


def plot(data: OrderedDict, title):
    lines = list(data.keys())
    reg_models = {d: log_regression(data[d]) for d in lines}
    for k in reg_models:
        print(k, reg_models[k])
    fig, ax = plt.subplots()

    # Add the mean trend lines
    for d, colour in zip(lines, COLOURS):
        assert isinstance(data[d], pd.DataFrame)
        accuracies = data[d]['acc'].values
        mean_trend = np.mean(accuracies.reshape(-1, NUM_REGRESSION_POINTS), axis=0)
        num_train = data[d]['num_train'][:NUM_REGRESSION_POINTS].values

        ax.loglog(num_train, mean_trend, marker='.', color=colour, linestyle='dashed')

    plt.legend(["{:11}slope={:.2f}, r={:.2f}".format(label, reg_models[label][0], reg_models[label][2]) for
                label in lines], prop={'family': 'monospace'}, loc="lower right")

    # Add the regression lines
    for d, colour in zip(lines, COLOURS):
        max_training = data[d]['num_train'].max()
        reg_line_x = np.linspace(1, max_training + 1)
        weight, bias = reg_models[d][:2]
        reg_line_y = weight * np.log(reg_line_x) + bias
        ax.plot(reg_line_x, reg_line_y, color=colour)

    ax.set_title(title)
    ax.set_xlabel("Number of subjects")
    ax.set_ylabel("Accuracy")
    ax.set_yticks([0.25, 0.4, 0.6, 0.8, 0.9])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_formatter(PercentFormatter(xmax=1.0))
    plt.show()
    fig.savefig(title+'.pdf', bbox_inches='tight')

    return reg_models


if __name__ == '__main__':
    to_load = OrderedDict()
    to_load['TIDNet+EA'] = 'DSCNN_{}_ea.xlsx'
    to_load['TIDNet'] = 'DSCNN_{}.xlsx'
    to_load['EEGNet+EA'] = 'EEGNet_{}_ea.xlsx'
    to_load['EEGNet'] = 'EEGNet_{}.xlsx'

    for targets in (2, 3, 4):
        print('Targets: ', targets)
        loaded = OrderedDict()
        for d in to_load:
            loaded[d] = pd.read_excel(TOPLEVEL / to_load[d].format(targets), header=1)
        models = plot(loaded, "MMI {}-way Classification".format(targets))
