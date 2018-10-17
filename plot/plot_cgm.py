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

def pad(xs, value=np.nan, maxlen=None):
    if maxlen is None:
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

def smooth_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
                                                                          mode='same')
    return xsmoo, ysmoo

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

def plot_mask_train_success():

    # Load all data.
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'mask_records.csv'))]
    # all_params = []

    var_param_keys = prepare_data(paths)

    data = {}
    max_len = 0
    # max_len_set = []
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'mask_records.csv'))
        if not results:
            print('skipping {}'.format(curr_path))

        print('loading {} ({})'.format(curr_path, len(results['episode'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)

        config = ''
        for k in var_param_keys:
            if k in params.keys():
                config += k + ": " + str(params[k])+"-"

        config = config[:-1]
        max_data_len = 20
        for k in results.keys():
            if "(r)" in k:
                this_set_name = config + "_" + k
                if len(results['episode']) > max_len:
                    max_len = len(results['episode'])
                    max_len_set = results['episode']
                if this_set_name not in data.keys():
                    data[this_set_name] = []
                if len(data[this_set_name]) < max_data_len:
                    data[this_set_name].append((results['episode'], results[k]))

    for k in data.keys():
        for i, r in enumerate(data[k]):
            vals = data[k][i][1]
            padding = np.ones((int(max_len) - len(data[k][i][1]))) * np.nan
            padded = np.concatenate([vals, padding], axis=0)
            data[k][i] = (data[k][i][0], padded)


    plt.clf()
    plt.figure(figsize=(9, 4.5))

    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    max_x = 150
    for config in sorted(data.keys()):
        # label = "mask " + config[1:-4] + " (n={})".format(len(data[config]))
        label = "mask " + config[1:-4]
        xs, ys = zip(*data[config])
        xs = np.array(xs)
        ys = np.array(ys)
        episodes_per_epoch = parallel_rollouts * training_rollout_cycles_per_epoch
        x_vals = max_len_set / episodes_per_epoch
        assert len(x_vals) == ys.shape[1]

        median = np.nanmedian(ys, axis=0)
        min_vals = np.nanpercentile(ys, 25, axis=0)
        max_vals = np.nanpercentile(ys, 75, axis=0)
        try:
            max_x_idx = min(np.argwhere(x_vals >= max_x))[0]
        except:
            print("Not all runs have {} epochs".format(max_x))
            max_x_idx = len(x_vals)
        median=median[:max_x_idx]
        min_vals=min_vals[:max_x_idx]
        max_vals = max_vals[:max_x_idx]
        x_vals = x_vals[:max_x_idx]

        if '000' in config:
            continue
            plt.plot(x_vals, median, label=label)
            plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
        else:
            _, smooth_median = smooth_curve(x_vals, median)
            _, smooth_min_vals = smooth_curve(x_vals, min_vals)
            _, smooth_max_vals = smooth_curve(x_vals, max_vals)
            plt.plot(x_vals, smooth_median, label=label)
            plt.fill_between(x_vals, smooth_min_vals, smooth_max_vals, alpha=0.25)

    ax = plt.gca()
    # ax.set_xlim([0, 17])
    ax.set_ylim([0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('success rate (training)')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.05,0.83))
    plt.legend(loc='upper left', bbox_to_anchor=(0.005,0.995))
    # plt.title("uniform sampling ($c=0$)", loc='right', pad=-80)
    # plt.title("title", loc='right', pad=-80)
    plt.savefig(os.path.join(args.dir, 'mask_train_success.png'))


def plot_subgoal_eval_success():

    # Load all data.
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'subgoal_records.csv'))]

    # var_param_keys = prepare_data(paths)

    subgoal_data = {}
    mask_data = {}
    max_len = 0
    n_goals = int((len(list(load_results(os.path.join(paths[0], 'subgoal_records.csv')).keys())) - 3) / 2)


    # max_len_set = []
    n_data = 0
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'subgoal_records.csv'))
        if not results:
            print('skipping {}'.format(curr_path))
        this_n_goals = int((len(list(results.keys()))-3)/2)
        if this_n_goals != n_goals:
            print("Error!!! number of goals not the same for all data")
            return
        print('loading {} ({})'.format(curr_path, len(results['episode'])))
        # with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        #     params = json.load(f)
        # config = ''
        # for k in var_param_keys:
        #     if k in params.keys():
        #         config += k + ": " + str(params[k])+"-"
        #
        # config = config[:-1]
        n_data += 1
        max_data_len = 200
        for k in results.keys():
            if "(r" in k:
                this_set_name = k
                if len(results['episode']) > max_len:
                    max_len = len(results['episode'])
                    max_len_set = results['episode']
                if this_set_name not in subgoal_data.keys():
                    subgoal_data[this_set_name] = []
                if len(subgoal_data[this_set_name]) < max_data_len:
                    subgoal_data[this_set_name].append((results['episode'], results[k]))

    for k in subgoal_data.keys():
        for i, r in enumerate(subgoal_data[k]):
            vals = subgoal_data[k][i][1]
            padding = np.ones((int(max_len) - len(subgoal_data[k][i][1]))) * np.nan
            padded = np.concatenate([vals, padding], axis=0)
            subgoal_data[k][i] = (subgoal_data[k][i][0], padded)


    plt.clf()
    plt.figure(figsize=(9, 4.5))
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    # max_x = 1000
    for config in sorted(subgoal_data.keys()):
        # label = "mask " + config[1:-4] + " (n={})".format(len(subgoal_data[config]))
        label = "subgoal " + config[0:-4]
        xs, ys = zip(*subgoal_data[config])
        xs = np.array(xs)
        ys = np.array(ys)
        # episodes_per_epoch = 4*64
        episodes_per_epoch = parallel_rollouts * eval_rollout_cycles_per_epoch
        x_vals = max_len_set / episodes_per_epoch
        assert len(x_vals) == ys.shape[1]

        median = np.nanmedian(ys, axis=0)
        min_vals = np.nanpercentile(ys, 25, axis=0)
        max_vals = np.nanpercentile(ys, 75, axis=0)
        # try:
        #     max_x_idx = min(np.argwhere(x_vals >= max_x))[0]
        # except:
        #     print("Not all runs have {} epochs".format(max_x))
        max_x_idx = len(x_vals)
        median = median[:max_x_idx]
        min_vals = min_vals[:max_x_idx]
        max_vals = max_vals[:max_x_idx]
        x_vals = x_vals[:max_x_idx]

        if '000' in config:
            plt.plot(x_vals, median, label=label)
            plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
        else:
            _, smooth_median = smooth_curve(x_vals, median)
            _, smooth_min_vals = smooth_curve(x_vals, min_vals)
            _, smooth_max_vals = smooth_curve(x_vals, max_vals)
            plt.plot(x_vals, smooth_median, label=label)
            plt.fill_between(x_vals, smooth_min_vals, smooth_max_vals, alpha=0.25)

    ax = plt.gca()
    # ax.set_xlim([0, max_x])
    ax.set_ylim([0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('estimated success rate (evaluation)')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.05,0.83))
    plt.legend(loc='upper left', bbox_to_anchor=(0.005, 0.995))
    # plt.title("uniform sampling ($c=0$)", loc='right', pad=-80)
    plt.title("title", loc='right', pad=-80)
    plt.savefig(os.path.join(args.dir, 'subgoal_eval_success.png'))


    # Draw estimated mask success probabilities
    epochs = max_len_set
    max_n = pow(2, n_goals)
    for n in range(max_n):
        mask_str = bin(n)[2:]
        mask_str = mask_str.rjust(n_goals, "0")
        # goal_masks.append(mask_str)
        mask_data[mask_str] = np.array([(epochs, np.ones(len(epochs)))] * n_data)

    for mask in sorted(mask_data.keys()):
        for sg_str, sgd in sorted(subgoal_data.items()):
            # print(sg_str)
            sg = int(sg_str[:-4])
            if mask[sg] == "1":
                for s_idx, sample in enumerate(sgd):
                    success_hist = sample[1]
                    mask_data[mask][s_idx][1] *= success_hist
                    # print('here')


    plt.clf()
    plt.figure(figsize=(9, 4.5))
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    # max_x = 1000
    for mask in sorted(mask_data.keys()):
        if mask == '000':
            continue
        # label = "mask " + config[1:-4] + " (n={})".format(len(subgoal_data[config]))
        label = "mask " + mask
        xs, ys = zip(*mask_data[mask])
        xs = np.array(xs)
        ys = np.array(ys)
        # episodes_per_epoch = 4*64
        episodes_per_epoch = parallel_rollouts*eval_rollout_cycles_per_epoch
        x_vals = max_len_set / episodes_per_epoch
        assert len(x_vals) == ys.shape[1]

        median = np.nanmedian(ys, axis=0)
        min_vals = np.nanpercentile(ys, 25, axis=0)
        max_vals = np.nanpercentile(ys, 75, axis=0)
        # try:
        #     max_x_idx = min(np.argwhere(x_vals >= max_x))[0]
        # except:
        #     print("Not all runs have {} epochs".format(max_x))
        max_x_idx = len(x_vals)
        median=median[:max_x_idx]
        min_vals=min_vals[:max_x_idx]
        max_vals = max_vals[:max_x_idx]
        x_vals = x_vals[:max_x_idx]

        if '000' in mask:
            plt.plot(x_vals, median, label=label)
            plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
        else:
            _, smooth_median = smooth_curve(x_vals, median)
            _, smooth_min_vals = smooth_curve(x_vals, min_vals)
            _, smooth_max_vals = smooth_curve(x_vals, max_vals)
            plt.plot(x_vals, smooth_median, label=label)
            plt.fill_between(x_vals, smooth_min_vals, smooth_max_vals, alpha=0.25)

    ax = plt.gca()
    ax.set_xlim([0, 150])
    ax.set_ylim([0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('estimated success rate (evaluation)')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.05,0.83))
    plt.legend(loc='upper left', bbox_to_anchor=(0.005,0.995))
    # plt.title("uniform sampling ($c=0$)", loc='right', pad=-80)
    # plt.title("title", loc='right', pad=-80)
    plt.savefig(os.path.join(args.dir, 'mask_eval_success_estimate.png'))


def plot_mask_frequency():

    # Load all data.
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'mask_records.csv'))]
    # all_params = []

    var_param_keys = prepare_data(paths)

    data = {}
    max_len = 0
    # max_len_set = []
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'mask_records.csv'),dtype=np.str)
        if not results:
            print('skipping {}'.format(curr_path))

        print('loading {} ({})'.format(curr_path, len(results['episode'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)

        config = ''
        for k in var_param_keys:
            if k in params.keys():
                config += k + ": " + str(params[k])+"-"

        config = config[:-1]
        episodes = results['episode']
        history_len = 40
        gm_hist = {}
        gm_means = {}
        for idx, gm in enumerate(results['goal mask']):
            if gm not in gm_hist.keys():
                gm_hist[gm] = deque(maxlen=history_len)
                gm_means[gm] = []
                # data[gm] = []
        for idx, gm in enumerate(results['goal mask']):
            if len(results['episode']) > max_len:
                max_len = len(results['episode'])
                max_len_set = results['episode']


            for gm2 in gm_hist.keys():
                if gm == gm2:
                    gm_hist[gm2].append(1)
                else:
                    gm_hist[gm2].append(0)
                gm_means[gm2].append(np.mean(gm_hist[gm2]))

        for gm in gm_means.keys():
            if gm not in data.keys():
                data[gm] = []
            data[gm].append((episodes, gm_means[gm]))

        max_len_set = np.array(max_len_set, dtype=np.int)

        for k in data.keys():
            for i, r in enumerate(data[k]):
                vals = data[k][i][1]
                padding = np.ones((int(max_len) - len(data[k][i][1]))) * np.nan
                padded = np.concatenate([vals, padding], axis=0)
                data[k][i] = (data[k][i][0], padded)


    plt.clf()
    plt.figure(figsize=(9, 4.5))

    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    # max_x = 50
    for mask in sorted(data.keys()):
        # label = "mask " + config[1:-4] + " (n={})".format(len(data[config]))
        label = "mask " + mask
        xs, ys = zip(*data[mask])
        xs = np.array(xs)
        ys = np.array(ys)
        episodes_per_epoch = parallel_rollouts * training_rollout_cycles_per_epoch
        x_vals = max_len_set / episodes_per_epoch
        assert len(x_vals) == ys.shape[1]

        median = np.nanmedian(ys, axis=0)
        min_vals = np.nanpercentile(ys, 25, axis=0)
        max_vals = np.nanpercentile(ys, 75, axis=0)
        # try:
        #     max_x_idx = min(np.argwhere(x_vals >= max_x))[0]
        # except:
        #     print("Not all runs have {} epochs".format(max_x))
        max_x_idx = len(x_vals)
        median=median[:max_x_idx]
        min_vals=min_vals[:max_x_idx]
        max_vals = max_vals[:max_x_idx]
        x_vals = x_vals[:max_x_idx]

        # if '000' in config:
        #     plt.plot(x_vals, median, label=label)
        #     plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
        # else:
        _, smooth_median = smooth_curve(x_vals, median)
        _, smooth_min_vals = smooth_curve(x_vals, min_vals)
        _, smooth_max_vals = smooth_curve(x_vals, max_vals)
        plt.plot(x_vals, smooth_median, label=label)
        plt.fill_between(x_vals, smooth_min_vals, smooth_max_vals, alpha=0.25)

    ax = plt.gca()
    # ax.set_xlim([0, max_x])
    # ax.set_ylim([0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('average frequency of mask use over last {} rollouts'.format(history_len*parallel_rollouts))
    # plt.legend(loc='upper left', bbox_to_anchor=(0.05,0.83))
    plt.legend(loc='upper left', bbox_to_anchor=(0.005,0.995))
    # plt.title("uniform sampling ($c=0$)", loc='right', pad=-80)
    plt.title("title", loc='right', pad=-80)
    plt.savefig(os.path.join(args.dir, 'mask_frequency.png'))

def plot_mask_probability():

    # Load all data.
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'mask_sample_prob.csv'))]
    # all_params = []

    var_param_keys = prepare_data(paths)

    rollouts_per_epoch = training_rollout_cycles_per_epoch * parallel_rollouts

    data = {}
    max_len = 0
    # max_len_set = []
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'mask_sample_prob.csv'))
        if not results:
            print('skipping {}'.format(curr_path))

        print('loading {} ({})'.format(curr_path, len(results[list(results.keys())[0]])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)

        config = ''
        for k in var_param_keys:
            if k in params.keys():
                config += k + ": " + str(params[k])+"-"

        plt.clf()
        plt.figure(figsize=(9, 4.5))

        plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
        # max_x = 50
        for gm in sorted(results.keys()):
            if gm == '':
                continue
            label = "mask " + gm
            ys = results[gm]

            xs = np.array(range(len(ys))) / rollouts_per_epoch
            ys = np.array(ys)

            _, smooth_ys = smooth_curve(xs, ys)
            plt.plot(xs, smooth_ys, label=label)

        ax = plt.gca()
        plt.xlabel('epoch')
        plt.ylabel('mask sampling probability')
        # plt.legend(loc='upper left', bbox_to_anchor=(0.05,0.83))
        plt.legend(loc='upper left', bbox_to_anchor=(0.005,0.995))
        # plt.title("uniform sampling ($c=0$)", loc='right', pad=-80)
        plt.title("title", loc='right', pad=-80)
        plt.savefig(os.path.join(args.dir, 'mask_sampling probability_{}.png'.format(config)))

def plot_subgoal_frequency():

    # Load all data.
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'mask_records.csv'))]
    # all_params = []

    var_param_keys = prepare_data(paths)

    data = {}
    max_len = 0
    # max_len_set = []
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        print('loading {}'.format(curr_path))
        results = load_results(os.path.join(curr_path, 'mask_records.csv'),dtype=np.str)
        if not results:
            print('skipping {}'.format(curr_path))

        print('loading {} ({})'.format(curr_path, len(results['episode'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)

        config = ''
        for k in var_param_keys:
            if k in params.keys():
                config += k + ": " + str(params[k])+"-"

        config = config[:-1]
        episodes = results['episode']
        history_len = 40
        gm_hist = {}
        gm_means = {}
        mask_size = len(results['goal mask'][0].strip())
        for gm in range(mask_size):
            if gm not in gm_hist.keys():
                gm_hist[gm] = deque(maxlen=history_len)
                gm_means[gm] = []
                # data[gm] = []
        for idx, gm in enumerate(results['goal mask']):
            gm = gm.strip()
            if len(results['episode']) > max_len:
                max_len = len(results['episode'])
                max_len_set = results['episode']


            for gm2 in gm_hist.keys():
                tmp = gm[gm2]
                if gm[gm2] == "1":
                    gm_hist[gm2].append(1)
                else:
                    gm_hist[gm2].append(0)
                gm_means[gm2].append(np.mean(gm_hist[gm2]))

        for gm in gm_means.keys():
            if gm not in data.keys():
                data[gm] = []
            data[gm].append((episodes, gm_means[gm]))

        max_len_set = np.array(max_len_set, dtype=np.int)

        for k in data.keys():
            for i, r in enumerate(data[k]):
                vals = data[k][i][1]
                padding = np.ones((int(max_len) - len(data[k][i][1]))) * np.nan
                padded = np.concatenate([vals, padding], axis=0)
                data[k][i] = (data[k][i][0], padded)


    plt.clf()
    plt.figure(figsize=(9, 4.5))

    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':']) * cycler('color', new_colors)))
    # max_x = 50
    for subgoal in sorted(data.keys()):
        # label = "mask " + config[1:-4] + " (n={})".format(len(data[config]))
        label = "subgoal " + str(subgoal)
        xs, ys = zip(*data[subgoal])
        xs = np.array(xs)
        ys = np.array(ys)
        episodes_per_epoch = parallel_rollouts*training_rollout_cycles_per_epoch
        x_vals = max_len_set / episodes_per_epoch
        assert len(x_vals) == ys.shape[1]

        median = np.nanmedian(ys, axis=0)
        min_vals = np.nanpercentile(ys, 25, axis=0)
        max_vals = np.nanpercentile(ys, 75, axis=0)
        # try:
        #     max_x_idx = min(np.argwhere(x_vals >= max_x))[0]
        # except:
        #     print("Not all runs have {} epochs".format(max_x))
        max_x_idx = len(x_vals)
        median=median[:max_x_idx]
        min_vals=min_vals[:max_x_idx]
        max_vals = max_vals[:max_x_idx]
        x_vals = x_vals[:max_x_idx]

        # if '000' in config:
        #     plt.plot(x_vals, median, label=label)
        #     plt.fill_between(x_vals, min_vals, max_vals, alpha=0.25)
        # else:
        _, smooth_median = smooth_curve(x_vals, median)
        _, smooth_min_vals = smooth_curve(x_vals, min_vals)
        _, smooth_max_vals = smooth_curve(x_vals, max_vals)
        plt.plot(x_vals, smooth_median, label=label)
        plt.fill_between(x_vals, smooth_min_vals, smooth_max_vals, alpha=0.25)

    ax = plt.gca()
    # ax.set_xlim([0, max_x])
    # ax.set_ylim([0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('average frequency of subgoal use over last {} rollouts'.format(history_len*parallel_rollouts))
    # plt.legend(loc='upper left', bbox_to_anchor=(0.05,0.83))
    plt.legend(loc='upper left', bbox_to_anchor=(0.005,0.995))
    # plt.title("uniform sampling ($c=0$)", loc='right', pad=-80)
    # plt.title("title", loc='right', pad=-80)
    plt.savefig(os.path.join(args.dir, 'subgoal_frequency.png'))

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
    # plot_mask_train_success()
    plot_subgoal_eval_success()
    # plot_mask_frequency()
    # plot_subgoal_frequency()
    # plot_mask_probability()

