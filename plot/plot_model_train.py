from plot.plot import *

def plot_model_train(data_dir):

    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/pred_err')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/pred_steps')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/loss-0')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/loss-0-grad')

    # lin_scale_graphs = ['train/loss-0', 'train/pred_err', 'train/pred_steps', 'train/loss-0-grad']
    # log_scale_graphs = ['train/loss-0', 'train/pred_err', 'train/loss-0-grad']
    # smoothen_graphs = ['train/pred_err', 'train/pred_steps']
    # mean_last = 10
    # results = load_results(progress_file)
    # datadir = os.path.dirname(progress_file)
    # if results is None:
    #     return
    # if 'epoch' not in results.keys():
    #     print("ERROR! Epoch is not a column in results file. Cannot draw plot.")
    #     return
    # xs = results['epoch']
    # for key, value in results.items():
    #     if key in lin_scale_graphs:
    #         plt.clf()
    #         fig = plt.figure(figsize=(10, 5))
    #         plt.plot(value)
    #
    #         if key in smoothen_graphs:
    #             smooth_x, smooth_y = smoothen_curve(xs,value)
    #             assert(smooth_x.all() == xs.all())
    #             plt.plot(smooth_y)
    #             plt.gca().annotate(str(smooth_y[-1]), (xs[-1], smooth_y[-1]))
    #         else:
    #             plt.gca().annotate(str(value[-1]), (xs[-1], value[-1]))
    #         plt.title(key)
    #         plt.savefig(datadir + "/lin_" + key.replace("/", "_") + ".png")
    #
    #     if key in log_scale_graphs:
    #         plt.clf()
    #         fig = plt.figure(figsize=(10, 5))
    #         plt.semilogy(value)
    #         if key in smoothen_graphs:
    #             smooth_x, smooth_y = smoothen_curve(xs,value)
    #             assert (smooth_x.all() == xs.all())
    #             plt.semilogy(smooth_y)
    #             plt.gca().annotate(str(smooth_y[-1]), (xs[-1], smooth_y[-1]))
    #         else:
    #             plt.gca().annotate(str(value[-1]), (xs[-1], value[-1]))
    #         plt.title(key)
    #         plt.savefig(datadir + "/log_" + key.replace("/", "_") + ".png")

