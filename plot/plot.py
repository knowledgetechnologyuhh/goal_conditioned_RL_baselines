import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import numpy as np
import json
import seaborn as sns;

sns.set()
import glob2
import argparse
from cycler import cycler


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
                                                                          mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


def draw_var_param_plots(data, var_param_keys, inter_dict, fig_dir,  y_axis_title=None):
    success_table = {}
    for config_split in var_param_keys:
        # config_split = 'mask_at_observation'
        for conf_val in inter_dict[config_split]:
            # print('processing {}: {}'.format(config_split, conf_val))
            # n_fig = 0
            plt.clf()
            # plt.figure(figsize=(20, 8))
            plt.figure(figsize=(9, 4.5))
            new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                          '#bcbd22', '#17becf']
            plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
            for ctr, config in enumerate(sorted(data.keys())):

                label = config + "-n: {}".format(len(data[config]))
                if "{}: {}-".format(config_split, conf_val) not in label:
                    continue
                # print('processing {}'.format(config))

                xs, ys = zip(*data[config])
                xs, ys = pad(xs), pad(ys, value=np.nan)
                median = np.nanmedian(ys, axis=0)
                mean = np.mean(ys, axis=0)
                assert xs.shape == ys.shape
                x_vals = range(1, xs.shape[1] + 1)

                plt.plot(x_vals, median, label=label)
                plt.fill_between(x_vals, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)

            plt.xlabel('Epoch')
            plt.ylabel(y_axis_title)
            plt.legend()
            plt.savefig(os.path.join(fig_dir, 'fig-{}:{}_{}.png'.format(config_split, conf_val, y_axis_title.replace("/", "_"))))


def draw_all_data_plot(data, fig_dir, y_axis_title=None, lin_log='lin'):
    plt.clf()
    # plt.figure(figsize=(20, 8))
    fig = plt.figure(figsize=(20, 8))
    ax = fig.gca()
    # new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    #               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    #               '#bcbd22', '#17becf']
    new_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    for idx, config in enumerate(sorted(data.keys(), reverse=True)):
        label = "+".join(sorted(config.split("-"), reverse=True))

        # Some custom modifications of label:
        label = label.replace("stochastic3_0_0_0_1", 'uniform')
        label = label.replace("stochastic3_0_0_0_1", 'uniform')
        label = label.replace("replay_k: 6", "DDPG+HER")
        label = label.replace("replay_k: 0", "DDPG")
        label = label.replace("+curriculum_sampling: none", '')
        if 'stochastic3_' in label:
            rg = label.split("stochastic3_")[1].split("_")[0]
            kappa = label.split("stochastic3_")[1].split("_")[2]
            h = label.split("stochastic3_")[1].split("_")[3].split("+")[0]
            gl_str = "curriculum_sampling: stochastic3_{}_0_{}_{}".format(rg,kappa,h)
            label = label.replace(gl_str, "CGM")
        label = label.replace("model_network_class: ", 'model: ')
        label = label.replace("baselines.model_based.model_rnn:", '')
        # End custom modifications of label

        xs, ys = zip(*data[config])
        xs, ys = pad(xs), pad(ys, value=np.nan)
        median = np.nanmedian(ys, axis=0)
        mean = np.mean(ys, axis=0)
        assert xs.shape == ys.shape
        x_vals = range(1,xs.shape[1]+1)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.xticks(loc=range(170))

        c_idx = idx % len(new_colors)
        color = new_colors[c_idx]

        if lin_log == 'lin':
            plt.plot(x_vals, median, label=label, color=color)
        elif lin_log == 'log':
            plt.semilogy(x_vals, median, label=label, color=color)

        plt.fill_between(x_vals, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color)

        # plt.plot(x_vals, mean, label=label+"-mean")
        # dev = np.std(ys, axis=0)
        # min_vals=mean-dev
        # max_vals = mean + dev
        # plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
    # plt.title(env_id)
    # ax.set_xlim([0, 150])
    plt.xlabel('epoch')
    plt.ylabel(y_axis_title)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_{}.png'.format(y_axis_title.replace("/", "_"))))


def draw_all_data_plot_rg_c_conv(data, fig_dir, y_axis_title=None):
    plt.clf()
    # plt.figure(figsize=(20, 8))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    for idx, config in enumerate(sorted(data.keys(), reverse=True)):
        label = config + " - n: {}".format(len(data[config]))
        label = config

        # Some custom modifications of label:
        label = label.replace("stochastic3_0_0_0_1", 'uniform')
        label = label.replace("stochastic3_0_0_0_1", 'uniform')
        label = label.replace("curriculum_sampling: none", 'no CGM')
        label = label.replace("curriculum_sampling: ", "")
        if 'stochastic3_' in label:
            rg = label.split("stochastic3_")[1].split("_")[0]
            kappa= label.split("stochastic3_")[1].split("_")[2]
            h = label.split("stochastic3_")[1].split("_")[3].split(" ")[0]
            # label = label.replace("stochastic3_{}_0_{}_{}".format(rg,c,h), 'rg={}, c={}'.format(rg,kappa))
            label = label.replace("stochastic3_{}_0_{}_{}".format(rg,kappa,h), 'CGM - $r_g$={}'.format(rg))
        # End custom modifications of label

        xs, ys = zip(*data[config])
        xs, ys = pad(xs), pad(ys, value=1.0)
        median = np.nanmedian(ys, axis=0)
        mean = np.mean(ys, axis=0)
        assert xs.shape == ys.shape
        x_vals = range(1,xs.shape[1]+1)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.xticks(loc=range(170))

        c_idx = idx % len(new_colors)
        color = new_colors[c_idx]

        plt.plot(x_vals, median, label=label, color=color)
        plt.fill_between(x_vals, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color)

        # plt.plot(x_vals, mean, label=label+"-mean")
        # dev = np.std(ys, axis=0)
        # min_vals=mean-dev
        # max_vals = mean + dev
        # plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
    # plt.title(env_id)
    ax.set_xlim([4, 15])
    plt.xlabel('epoch')
    plt.ylabel(y_axis_title)
    plt.legend(loc='upper left')
    # fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig.png'))

def draw_stochastic_surface_plot(data, percent_to_achieve, fig_dir):
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
    plt.savefig(os.path.join(fig_dir, 'c_rg_.png'))


def get_var_param_keys(paths):
    # all_params = []
    inter_dict = {}
    var_param_keys = set()
    max_epochs = 0
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        # print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            # print('skipping plot for {}'.format(curr_path))
            continue
        # if len(results) != 10:
        #     print('skipping {}, bad file format'.format(curr_path))
        #     continue
        # print('loading {} ({})'.format(curr_path, len(results['epoch'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)
        for k, v in params.items():
            if k not in inter_dict.keys():
                inter_dict[k] = [v]
            if v not in inter_dict[k]:
                inter_dict[k].append(v)
                var_param_keys.add(k)
        max_epochs = max(max_epochs, len(results['epoch']))
    return var_param_keys, inter_dict, max_epochs

def get_data(paths, var_param_keys, max_epochs, smoothen=False, padding=True, col_to_display='test/success_rate', data_lastval_threshold=0.0):
    data = {}
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        # print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            # print('skipping plot for {}'.format(curr_path))
            continue
        # print('loading {} ({})'.format(curr_path, len(results['epoch'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)
        if col_to_display not in results.keys():
            continue
        this_data = np.array(results[col_to_display])

        epoch = np.array(results['epoch']) + 1
        if len(epoch) < 3:
            continue
        env_id = params['env_name']

        config = ''
        for k in var_param_keys:
            if k in params.keys():
                config += k + ": " + str(params[k])+"-"
        config = config[:-1]

        # Process and smooth data.
        assert this_data.shape == epoch.shape
        x = epoch
        y = np.array(this_data)
        if padding:
            x = np.array(range(1, max_epochs+1))
            pad_val = y[-1]
            y_pad = np.array([pad_val] * (max_epochs - len(y)))
            y = np.concatenate((y,y_pad))

        if smoothen:
            x, y = smooth_reward_curve(epoch, this_data)
        if x.shape != y.shape:
            continue

        if y[-1] >= data_lastval_threshold or (len(epoch) == max_epochs):
            if config not in data.keys():
                data[config] = []
            data[config].append((x, y))

    return data


def get_best_data(data, sort_order, n_best=5, avg_last_steps=5):
    d_keys = [key for key in sorted(data.keys())]
    if sort_order == 'max':
        best_vals = [0 for _ in range(n_best)]
    elif sort_order in ['min', 'least']:
        best_vals = [np.iinfo(np.int16).max for _ in range(n_best)]
    best_keys = ['' for _ in range(n_best)]
    for key in d_keys:
        if sort_order in ['max', 'min']:
            last_vals = np.array([data[key][i][1][-avg_last_steps:] for i in range(len(data[key])) if
                                  len(data[key][i][1]) > avg_last_steps])
            last_n_avg = np.mean(last_vals)
            if sort_order == 'max':
                if last_n_avg > np.min(best_vals):
                    ri = np.argmin(best_vals)
                    best_vals[ri] = last_n_avg
                    best_keys[ri] = key
            elif sort_order == 'min':
                if last_n_avg < np.max(best_vals):
                    ri = np.argmax(best_vals)
                    best_vals[ri] = last_n_avg
                    best_keys[ri] = key
        elif sort_order == 'least':
            all_lens = np.array([max(data[key][i][0]) for i in range(len(data[key]))])
            last_n = np.mean(all_lens)
            if last_n < np.max(best_vals):
                ri = np.argmax(best_vals)
                best_vals[ri] = last_n
                best_keys[ri] = key
    best_data = {}
    for key in best_keys:
        if key in data.keys():
            best_data[key] = data[key]
    return best_data


def do_plot(data_dir, smoothen=True, padding=False, col_to_display='test/success_rate', get_best='least', lin_log='lin'):
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams['font.weight'] = 'normal'
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(data_dir, '**', 'progress.csv'))]
    var_param_keys, inter_dict, max_epochs = get_var_param_keys(paths)
    data = get_data(paths, var_param_keys, max_epochs, smoothen, padding, col_to_display=col_to_display)
    if get_best != '':
        data = get_best_data(data, get_best, n_best=10, avg_last_steps=5)
    draw_all_data_plot(data, data_dir, y_axis_title=col_to_display, lin_log=lin_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--column', type=str, default='test_0/success_rate')
    args = parser.parse_args()
    # data_lastval_threshold = 0.0
    do_plot(args.data_dir, args.smooth, args.pad, col_to_display=args.column)