# from plot import do_plot
import plot
import argparse


def plot_model_train(data_dir):
    # plt.switch_backend('agg')

    # define plot key, sort by min or max, linear or logarithmic scale
    plot_keys = [
        ('train/observation loss', 'min', 'lin'),
        ('train/loss prediction loss', 'min', 'log'),
        ('train/pred_err', 'min', 'lin'),
        ('train/pred_steps', 'max', 'lin'),
        ('train/loss_pred_steps', 'max', 'lin'),
        ('train/mj_pred_err', 'min', 'lin'),
        ('train/mj_pred_steps', 'max', 'lin'),
        ('train/mj acc. err', 'min', 'lin'),
        ('train/acc. err', 'min', 'lin'),
        ('train/variance_div_acc_err', 'max', 'lin'),
        ('train/buffer observation variance', 'max', 'lin'),
        ('mb_policy/model_lr', 'min', 'lin')
    ]
    for key in plot_keys:
        plot.do_plot(data_dir, smoothen=False, padding=False, col_to_display=key[0], get_best=key[1], lin_log=key[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    plot_model_train(data_dir=args.data_dir)
