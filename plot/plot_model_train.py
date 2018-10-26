from plot.plot import *

def plot_model_train(data_dir):
    # plt.switch_backend('agg')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/pred_err')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/pred_steps')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/loss-0')
    do_plot(data_dir, smoothen=True, padding=False, col_to_display='train/loss-0-grad')

