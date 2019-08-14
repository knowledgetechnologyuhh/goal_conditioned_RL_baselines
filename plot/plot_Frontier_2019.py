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


def draw_all_data_plot(data, fig_dir, x_axis_title=None, y_axis_title=None, lin_log='lin'):
    plt.clf()
    # plt.figure(figsize=(20, 8))
    fig = plt.figure(figsize=(20, 8))
    ax = fig.gca()
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    # new_colors = sorted(plt.rcParams['axes.prop_cycle'].by_key()['color'], reverse=True)
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))

    temp = sorted(data.keys(), reverse=False)
    # this is to solve the color mismatch of coeff 0.06 in 1 block tower
    try:
        temp.append(temp.pop(temp.index('obs_noise_coeff: 0.06')))
    except Exception as e:
        pass
    for idx, config in enumerate(temp):
        label = "|".join(sorted(config.split("|"), reverse=True))

        # Some custom modifications of label:
        label = label.replace("model_network_class: ", 'model: ')
        label = label.replace("baselines.model_based.model_rnn:", '')
        label = label.replace("algorithm: baselines.herhrl", 'PDDL+HER')
        label = label.replace("algorithm: baselines.her", 'HER')
        label = label.replace("obs_noise_coeff:", 'obs noise:')
        # End custom modifications of label

        xs, ys = zip(*data[config])
        # label = label + "|N:{}".format(len(ys))
        len_ys = sorted([len(y) for y in ys])
        # maxlen_ys_2nd = len_ys[-2]
        maxlen_ys = max([len(x) for x in xs])
        ys = pad(ys, value=np.nan)
        x_max = [max(x) for x in xs]
        x_vals = xs[np.argmax(x_max)]
        # xs, ys = pad(xs), pad(ys, value=np.nan)
        median = np.nanmedian(ys, axis=0)
        mean = np.mean(ys, axis=0)
        # assert xs.shape == ys.shape
        # x_vals = range(1,xs.shape[1]+1)
        # x_vals = xs

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        c_idx = idx % len(new_colors)
        color = new_colors[c_idx]
        plot_idx = maxlen_ys
        if lin_log == 'lin':
            plt.plot(x_vals[:plot_idx], median[:plot_idx], label=label, color=color)
        elif lin_log == 'log':
            plt.semilogy(x_vals, median, label=label, color=color)
        plt.fill_between(x_vals[:plot_idx],
                         np.nanpercentile(ys, 25, axis=0)[:plot_idx],
                         np.nanpercentile(ys, 75, axis=0)[:plot_idx], alpha=0.25, color=color)

        # plt.plot(x_vals, mean, label=label+"-mean")
        # dev = np.std(ys, axis=0)
        # min_vals=mean-dev
        # max_vals = mean + dev
        # plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
    # plt.title(env_id)
    # ax.set_xlim([0, 150])
    ax.tick_params(labelsize=20)
    y_axis_display_title = y_axis_title.replace("/", " ")
    y_axis_display_title = y_axis_display_title.replace("_", " ")
    plt.xlabel(x_axis_title, fontsize=20)
    plt.ylabel(y_axis_display_title, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    if len(labels) > 0:
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        plt.legend(handles, labels, loc='upper left', fontsize=16)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_{}.jpg'.format(y_axis_title.replace("/", "_"))))
    plt.savefig(os.path.join(fig_dir, 'fig_{}.pdf'.format(y_axis_title.replace("/", "_"))))



def get_var_param_keys(paths, x_vals):
    # all_params = []
    inter_dict = {}
    var_param_keys = set()
    max_x = 0
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
        max_x = max(max_x, max(results[x_vals]))
    return var_param_keys, inter_dict, max_x

def get_data(paths, var_param_keys, max_x, x_vals, smoothen=False, padding=True, col_to_display='test/success_rate'):
    data = {}
    max_x_idx = {}
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

        xs = np.array(results[x_vals])

        if len(xs) < 3:
            continue

        config = ''
        for k in var_param_keys:
            if k in params.keys():
                config += k + ": " + str(params[k])+"|"
        config = config[:-1]

        # Process and smooth data.
        assert this_data.shape == xs.shape
        x = xs
        y = np.array(this_data)
        if padding:
            print("Padding currently not supported")
            raise NotImplementedError
            #TODO: implement extrapolation of x-values.
            # x = np.array(range(max_x))
            # pad_val = y[-1]
            # y_pad = np.array([pad_val] * (max_x - len(y)))
            # y = np.concatenate((y,y_pad))

        if smoothen:
            x, y = smooth_reward_curve(xs, this_data)
        if x.shape != y.shape:
            continue
        max_x_idx[config] = int(max(np.argwhere(xs <= max_x)))
        if config not in data.keys():
            data[config] = []
        data[config].append((x, y))
    cut_data = {}
    for k,v in data.items():
        cut_data[k] = []
        for idx,d in enumerate(data[k]):
            cut_data[k].append([data[k][idx][0][:max_x_idx[k]]])
            cut_data[k][-1].append(data[k][idx][1][:max_x_idx[k]])
            # data[k][idx][0] = data[k][idx][0][:max_epochs]
            # data[k][idx][1] = data[k][idx][1][:max_epochs]

    return cut_data


def get_best_data(data, sort_order, n_best=5, avg_last_steps=5, sort_order_least_val=None):
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
            if sort_order_least_val is not None:
                all_lens = np.array(
                    [np.argwhere(data[key][i][1] >= sort_order_least_val) for i in range(len(data[key]))]).squeeze(2)
                all_lens = [min(lens) for lens in all_lens]
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


def get_min_len_data(data, min_len):
    d_keys = [key for key in sorted(data.keys())]
    new_data = {}
    for key in d_keys:
        new_data[key] = []
        for d in data[key]:
            if np.max(d[0]) >= min_len:
                new_data[key].append(d)
        if len(new_data[key]) == 0:
            new_data.pop(key)
    return new_data

def get_paths_with_symlinks(data_dir, maxdepth=8):
    glob_path = data_dir
    paths = []
    for _ in range(maxdepth):
        path_to_use = os.path.join(glob_path, 'progress.csv')
        paths += [os.path.abspath(os.path.join(path, '..')) for path in
                  glob2.glob(path_to_use)]
        glob_path = os.path.join(glob_path, '*')
    return paths

def do_plot(data_dir, x_vals='epoch', smoothen=True, padding=False, col_to_display='test/success_rate', get_best='least', lin_log='lin', min_len=10, cut_early_x=None):
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams['font.weight'] = 'normal'
    paths = get_paths_with_symlinks(data_dir, maxdepth=8)
    # paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(data_dir, '**', 'progress.csv'))]
    var_param_keys, inter_dict, max_x = get_var_param_keys(paths, x_vals)
    if cut_early_x is not None:
        max_x = min(max_x, cut_early_x)
    try:
        var_param_keys.remove('base_logdir')
    except Exception as e:
        pass
    if 'algorithm' in var_param_keys:
        if 'obs_noise_coeff' in var_param_keys:
            var_param_keys = set()
            var_param_keys.add('algorithm')
            var_param_keys.add('obs_noise_coeff')
            # var_param_keys.add('n_episodes')
        else:
            var_param_keys = set()
            var_param_keys.add('algorithm')
    if 'early_stop_success_rate' in var_param_keys:
        var_param_keys.remove('early_stop_success_rate')
    data = get_data(paths, var_param_keys, max_x, x_vals=x_vals, smoothen=smoothen, padding=padding, col_to_display=col_to_display)
    data = get_min_len_data(data, min_len=min_len)
    # if get_best != '':
    #     data = get_best_data(data, get_best, n_best=10, avg_last_steps=5, sort_order_least_val=0.5)
    # try:
    #     draw_all_data_plot(data, data_dir, x_axis_title=x_vals, y_axis_title=col_to_display, lin_log=lin_log)
    # except Exception as e:
    #     print("This does not work for some reason: {}".format(e))
    draw_all_data_plot(data, data_dir, x_axis_title=x_vals, y_axis_title=col_to_display, lin_log=lin_log)
    # draw_all_data_plot(data, data_dir, y_axis_title=col_to_display, lin_log=lin_log)


def get_all_columns(data_dir, exclude_cols=['epoch','rollouts', 'steps', 'buffer_size']):
    cols = []
    paths = get_paths_with_symlinks(data_dir, maxdepth=8)
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        # print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'progress.csv'))
        cols += results.keys()
    cols = list(set(cols))
    ret_cols = []
    for c in cols:
        remove = False
        for ex_c in exclude_cols:
            if c.find(ex_c) != -1:
                remove = True
        if remove == False:
            ret_cols.append(c)
    return ret_cols

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--column', type=str, default='')
    parser.add_argument('--cut_early_x', type=int, default=None)
    parser.add_argument('--x_vals', type=str, default='epoch', choices=['epoch', 'train/rollouts'])
    args = parser.parse_args()
    cols = get_all_columns(args.data_dir)
    if args.column == '':
        for c in cols:
            do_plot(args.data_dir, args.x_vals, args.smooth, args.pad, col_to_display=c, cut_early_x=args.cut_early_x)
    else:
    # data_lastval_threshold = 0.0
        do_plot(args.data_dir, args.x_vals, args.smooth, args.pad, col_to_display=args.column, cut_early_x=args.cut_early_x)