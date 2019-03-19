import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns;
from collections import deque
sns.set()
import glob2
import argparse
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.ticker import MaxNLocator


def load_results(file, dtype=None):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    if dtype is None:
        data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    else:
        data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0., dtype=dtype)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result

# def pad(xs, value=np.nan, maxlen=None):
#     if maxlen is None:
#         maxlen = np.max([len(x) for x in xs])
#
#     padded_xs = []
#     for x in xs:
#         if x.shape[0] >= maxlen:
#             padded_xs.append(x)
#
#         padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
#         x_padded = np.concatenate([x, padding], axis=0)
#         assert x_padded.shape[1:] == x.shape[1:]
#         assert x_padded.shape[0] == maxlen
#         padded_xs.append(x_padded)
#     return np.array(padded_xs)
#
# def smooth_curve(x, y):
#     halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
#     k = halfwidth
#     xsmoo = x
#     ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
#                                                                           mode='same')
#     return xsmoo, ysmoo

def prepare_data(paths):
    inter_dict = {}
    var_param_keys = set()
    max_episodes = 0
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        print('loading {}'.format(curr_path))
        # results = load_results(os.path.join(curr_path, 'mask_records.csv'))
        # if not results:
        #     print('skipping {}'.format(curr_path))
        #     continue

        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)
        for k,v in params.items():
            if k not in inter_dict.keys():
                inter_dict[k] = [v]
            if v not in inter_dict[k]:
                inter_dict[k].append(v)
                var_param_keys.add(k)
        # max_episodes = max(max_episodes, len(results['episode']))
    return var_param_keys

def plot_epochs_success(data, percent_to_achieve, fig_dir):
    plt.clf()
    # fig = plt.figure(figsize=(20, 8))
    plt.figure(figsize=(9, 4.5))
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    # plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    surf_plot_data = {}
    uniform_sampling_epochs = []
    none_sampling_epochs = []
    kappa_s = set()
    rg_s = set()
    for config in sorted(data.keys()):

        epochs = []
        for d in data[config]:
            try:
                epoch = min(np.argwhere(d[1] > percent_to_achieve))[0]
            except:
                print("Not enough data for {}".format(config))
                continue
            epochs.append(epoch)
        # epochs = [len(d[0]) for d in data[config]]
        if 'curriculum_sampling: none' in config:
            none_sampling_epochs += epochs
            kappa_s.add(-1)
            continue

        median_epochs = np.median(epochs)
        min_perc = np.nanpercentile(epochs, 25, axis=0)
        max_perc = np.nanpercentile(epochs, 75, axis=0)
        avg_epochs = np.mean(epochs)
        n_runs = len(epochs)
        std_epochs = np.std(epochs)

        if 'stochastic3_' not in config:
            continue

        rg = float(config.split("stochastic3_")[1].split("_")[0])
        rg_s.add(rg)
        kappa = float(config.split("stochastic3_")[1].split("_")[2])
        kappa_s.add(kappa)

        if rg not in surf_plot_data.keys():
            surf_plot_data[rg] = {}
        if kappa == 0.0:
            uniform_sampling_epochs += epochs
        surf_plot_data[rg][kappa] = (avg_epochs, std_epochs, n_runs, median_epochs, min_perc, max_perc)

    uniform_avg_epochs = np.mean(uniform_sampling_epochs)
    none_avg_epochs = np.mean(none_sampling_epochs)
    uniform_std_epochs = np.std(uniform_sampling_epochs)
    none_std_epochs = np.std(none_sampling_epochs)
    uniform_median_epochs = np.median(uniform_sampling_epochs)
    none_median_epochs = np.median(none_sampling_epochs)
    uniform_min_perc = np.nanpercentile(uniform_sampling_epochs, 25, axis=0)
    none_min_perc = np.nanpercentile(none_sampling_epochs, 25, axis=0)
    uniform_max_perc = np.nanpercentile(uniform_sampling_epochs, 75, axis=0)
    none_max_perc = np.nanpercentile(none_sampling_epochs, 75, axis=0)
    for rg in surf_plot_data.keys():
        surf_plot_data[rg][0.0] = (
        uniform_avg_epochs, uniform_std_epochs, len(uniform_sampling_epochs), uniform_median_epochs, uniform_min_perc,
        uniform_max_perc)
        surf_plot_data[rg][-1] = (
        none_avg_epochs, none_std_epochs, len(none_sampling_epochs), none_median_epochs, none_min_perc,
        none_max_perc)


    kappa_s = sorted(list(kappa_s))
    # kappa_s.insert(1,0)
    rg_s = sorted(list(rg_s))
    # surf_plot_data_arr = np.array(list(surf_plot_data.items()))
    for idx, kappa in enumerate(kappa_s):
        # label = "c={} -n: {}".format(c, len(surf_plot_data[0][kappa]))
        # n_runs = ''
        # n_runs = np.mean(0)
        c_label = "$\kappa$={}".format(kappa)
        if kappa== -1:
            c_label = "no CGM"
            continue
        if kappa== 0:
            c_label = "uniform GM"
            continue
        label = "{}".format(c_label)
        xs = sorted(list(surf_plot_data.keys()))
        xs = np.array([k for k in sorted(surf_plot_data.keys()) if kappa in surf_plot_data[k].keys()])

        # ys = np.array([surf_plot_data[k][kappa][0] for k in sorted(surf_plot_data.keys()) if kappain surf_plot_data[k].keys()])
        # std_ys = np.array([surf_plot_data[k][kappa][1] for k in sorted(surf_plot_data.keys()) if kappa in surf_plot_data[k].keys()])
        # min_vals = ys + std_ys
        # max_vals = ys - std_ys

        ys = np.array([surf_plot_data[k][kappa][3] for k in sorted(surf_plot_data.keys()) if kappa in surf_plot_data[k].keys()])
        n_runs = np.array([surf_plot_data[k][kappa][2] for k in sorted(surf_plot_data.keys()) if kappa in surf_plot_data[k].keys()])
        min_vals = np.array([surf_plot_data[k][kappa][4] for k in sorted(surf_plot_data.keys()) if kappa in surf_plot_data[k].keys()])
        max_vals = np.array([surf_plot_data[k][kappa][5] for k in sorted(surf_plot_data.keys()) if kappa in surf_plot_data[k].keys()])

        if np.array(xs).shape != ys.shape:
            print("This data probably has not all kappas")
            continue

        color = new_colors[idx]

        print("C {} has color {}".format(kappa,color))

        # Add median points
        plt.scatter(xs, ys, color=color)
        # Add number of runs
        # for d_idx, n in enumerate(n_runs):
        #     plt.gca().annotate(str(n), (xs[d_idx], ys[d_idx]))
        # Add lines
        plt.plot(xs, ys, label=label, color=color)
        # Add quartiles
        plt.plot(xs, min_vals, linestyle='dashed', color=color, alpha=0.25)
        plt.plot(xs, max_vals, linestyle='dashed', color=color, alpha=0.25)
        # break
        # plt.fill_between(xs, min_vals, max_vals, alpha=0.25)
        # plt.fill_between(xs, min_vals, max_vals, alpha=0.1)
    # plt.legend(loc='upper left', bbox_to_anchor=(5.05,1.83))
    ax = plt.gca()
    # ax.set_xlim([0, 70])
    ax.set_ylim([20, 80])
    plt.xlabel('$c_g$')
    plt.ylabel('epochs to achieve {}% success rate'.format(int(percent_to_achieve*100)))

    plt.legend(loc='upper left')
    # plt.title("Number of epochs to achieve {}% success rate".format(int(percent_to_achieve*100)), loc='center', pad=-20)
    plt.savefig(os.path.join(fig_dir, 'penalty_hyperopt_.png'))

if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams['font.weight'] = 'normal'
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    parallel_rollouts=4
    training_rollout_cycles_per_epoch=64
    eval_rollout_cycles_per_epoch = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--smooth', type=int, default=1)
    args = parser.parse_args()
    plot_epochs_success(data, 50, parser.args.dir)

