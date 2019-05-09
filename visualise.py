import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style
sns.set(context='talk', font_scale=0.9)
sns.set_style('white')


def input_vs_output_data(data_X, data_y, fig, reduced_plot=True,
                         xlims=(-3e-4, 3e-4), ylims=(-1e-5, 1e-5),
                         units=True):
    """ Function to visualise training data (input and output).
    reduced_plot: if True not all particles will be shown. """
    # TODO: make more generic for any training data

    axs = [fig.add_subplot(121), fig.add_subplot(122)]

    if reduced_plot:
        step = int(data_X.shape[0] / 1000.)
    else:
        step = 1

    data_X.iloc[::step].plot.scatter(
        x='x', y='xp', ax=axs[0], s=14, c='darkred', alpha=0.4,
        label='Initial', linewidths=0, legend=None)
    data_y.iloc[::step].plot.scatter(
        x='x', y='xp', ax=axs[0], s=14, c='mediumblue', alpha=0.4,
        label='Target', linewidths=0, legend=None)
    data_X.iloc[::step].plot.scatter(
        x='y', y='yp', ax=axs[1], s=14, c='darkred', alpha=0.4,
        label='Initial', linewidths=0)
    data_y.iloc[::step].plot.scatter(
        x='y', y='yp', ax=axs[1], s=14, c='mediumblue', alpha=0.4,
        label='Target', linewidths=0)

    axs[0].set_title('Horizontal plane')
    axs[1].set_title('Vertical plane')
    if units:
        axs[0].set(xlabel="x (m)", ylabel="x'")
        axs[1].set(xlabel="y (m)", ylabel="y'")
    else:
        axs[0].set(xlabel="x", ylabel="x'")
        axs[1].set(xlabel="y", ylabel="y'")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.02))

    for ax in axs:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0),
                            useMathText=True)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplots_adjust(
        left=0.09, bottom=0.12, top=0.81, right=0.83, wspace=0.34)
    plt.show()


def test_data_phase_space(prediction, ground_truth, fig,
                          xlims=(-3e-4, 3e-4), ylims=(-1e-5, 1e-5),
                          units=True):
    axs = [fig.add_subplot(121), fig.add_subplot(122)]

    ground_truth.plot.scatter(
        x='x', y='xp', ax=axs[0], s=70, c='darkred',
        label='Ground truth', linewidths=0, legend=None)
    prediction.plot.scatter(
        x='x', y='xp', ax=axs[0], s=40, c='cornflowerblue',
        marker='x', label='Prediction', linewidths=1, legend=None)
    ground_truth.plot.scatter(
        x='y', y='yp', ax=axs[1], s=70, c='darkred',
        label='Ground truth', linewidths=0)
    prediction.plot.scatter(
        x='y', y='yp', ax=axs[1], s=40, c='cornflowerblue',
        marker='x', label='Prediction', linewidths=1)

    axs[0].set_title('Horizontal plane')
    axs[1].set_title('Vertical plane')
    if units:
        axs[0].set(xlabel="x (m)", ylabel="x'")
        axs[1].set(xlabel="y (m)", ylabel="y'")
    else:
        axs[0].set(xlabel="x", ylabel="x'")
        axs[1].set(xlabel="y", ylabel="y'")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.02))

    for ax in axs:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0),
                            useMathText=True)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplots_adjust(
        left=0.09, bottom=0.12, top=0.81, right=0.8, wspace=0.34)
    plt.show()

def test_data_phase_space_single(
        prediction, ground_truth, fig, xlims=(-1.2e-4, 1.2e-4),
        ylims=(-1.2e-6, 1.2e-6), units=True, txt=""):
    axs = [fig.add_subplot(111)]

    ground_truth.plot.scatter(
        x='y', y='yp', ax=axs[0], s=70, c='tab:green',
        label='Ground truth\n(full tracking)', linewidths=0)
    prediction.plot.scatter(
        x='y', y='yp', ax=axs[0], s=50, c='tab:red',
        marker='x', label='NN prediction', linewidths=0.5)

    # axs[0].set_title('Vertical plane')
    if units:
        axs[0].set(xlabel=r"$y$ (m)", ylabel=r"$y'$")
    else:
        axs[0].set(xlabel=r"$y$", ylabel=r"$y'$")
    axs[0].legend(fontsize=15)

    for ax in axs:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0),
                            useMathText=True)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    axs[0].text(0.05, 0.06, s=txt, horizontalalignment='left',
                transform=axs[0].transAxes)

    plt.subplots_adjust(
        left=0.16, bottom=0.12, top=0.86, right=0.92, wspace=0.34)
    plt.show()


def test_data(prediction, ground_truth, fig, xlabel='Particle ID'):
    axs = [fig.add_subplot(221), fig.add_subplot(222),
           fig.add_subplot(223), fig.add_subplot(224)]

    ground_truth.plot(
        y='x', ax=axs[0], c='darkred', label='Ground truth',
        legend=None)
    prediction.plot(y='x', ax=axs[0], ls='--', c='cornflowerblue',
                    label='Prediction', legend=None)
    ground_truth.plot(y='xp', ax=axs[2], c='darkred', legend=None)
    prediction.plot(
        y='xp', ax=axs[2], ls='--', c='cornflowerblue', legend=None)

    ground_truth.plot(
        y='y', ax=axs[1], c='darkred', label='Ground truth')
    prediction.plot(y='y', ax=axs[1], ls='--', c='cornflowerblue',
                    label='Prediction')
    ground_truth.plot(y='yp', ax=axs[3], c='darkred', legend=None)
    prediction.plot(
        y='yp', ax=axs[3], ls='--', c='cornflowerblue', legend=None)

    axs[0].set_title('Horizontal plane')
    axs[1].set_title('Vertical plane')
    axs[0].set(ylabel="x (m)")
    axs[1].set(ylabel="y (m)")
    axs[2].set(xlabel=xlabel, ylabel="x'")
    axs[3].set(xlabel=xlabel, ylabel="y'")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.03, 1.02))

    for ax in axs:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
        ax.set_xlim(0, prediction.shape[0])
        # ax.set_ylim(ylims)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.legend()

    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.setp(axs[1].get_xticklabels(), visible=False)
    plt.subplots_adjust(
        left=0.08, bottom=0.12, top=0.84, right=0.83, wspace=0.27)
    plt.show()


def test_data_difference(difference, fig, xlabel='Particle ID'):
    axs = [fig.add_subplot(211), fig.add_subplot(212)]

    difference.plot(
        y=['x', 'y'], ax=axs[0], color=["peru", "forestgreen"])
    difference.plot(
        y=['xp', 'yp'], ax=axs[1], color=["peru", "forestgreen"])
    axs[0].set(ylabel=r"$\Delta u$")
    axs[1].set(xlabel=xlabel, ylabel=r"$\Delta u'$")
    axs[0].legend([r'$x$', r'$y$'])
    axs[1].legend([r"$x'$", r"$y'$"])

    for ax in axs:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
        ax.set_xlim(0, difference.shape[0])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        lim_max = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-lim_max, lim_max)

    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.subplots_adjust(left=0.14, bottom=0.12, top=0.84, right=0.83)
    plt.show()


def training_evolution_NN(history, fig):
    axs = [fig.add_subplot(211), fig.add_subplot(212)]

    history.plot(
        y=['loss', 'val_loss'], ax=axs[0], color=["r", "b"])
    history.plot(
        y=['acc', 'val_acc'], ax=axs[1], color=["r", "b"])
    axs[0].set(ylabel="Loss")
    axs[0].set_yscale("log")
    axs[1].set(xlabel="Epoch", ylabel="Accuracy")
    axs[0].legend(["Training", "Validation"])
    axs[1].legend(["Training", "Validation"])

    for ax in axs:
        ax.set_xlim(0, history.shape[0])
        # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)

    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.subplots_adjust(left=0.14, bottom=0.12, top=0.84, right=0.83)
    plt.show()