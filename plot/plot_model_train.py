# from plot import do_plot
import plot
import argparse

def plot_model_train(data_dir):
    # plt.switch_backend('agg')
    plot_keys = ['train/total loss', 'train/observation loss', 'train/loss prediction loss',
                 'train/pred_err', 'train/pred_steps', 'train/loss_pred_steps',
                 'train/mj_pred_err', 'train/mj_pred_steps']
    for key in plot_keys:
        plot.do_plot(data_dir, smoothen=False, padding=False, col_to_display=key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    plot_model_train(data_dir=args.data_dir)

