import matplotlib.pyplot as plt


def phase_space_data(data_X, data_y, fig, reduced_plot=True,
                     xlims=(-3e-4, 3e-4), ylims=(-1e-5, 1e-5)):
    """ Function to visualise training data (input and output).
    reduced_plot: if True not all particles will be shown. """
    # TODO: make more generic for any training data

    axs0 = [fig.add_subplot(121), fig.add_subplot(122)]

    if reduced_plot:
        step = int(data_X.shape[0] / 1000.)
    else:
        step = 1

    data_X.iloc[::step].plot.scatter(
        x='x', y='xp', ax=axs0[0], s=14, c='darkred', alpha=0.4,
        label='Input', linewidths=0)
    data_y.iloc[::step].plot.scatter(
        x='x', y='xp', ax=axs0[0], s=14, c='mediumblue', alpha=0.4,
        label='Target', linewidths=0)
    data_X.iloc[::step].plot.scatter(
        x='y', y='yp', ax=axs0[1], s=14, c='darkred', alpha=0.4,
        label='Input', linewidths=0)
    data_y.iloc[::step].plot.scatter(
        x='y', y='yp', ax=axs0[1], s=14, c='mediumblue', alpha=0.4,
        label='Target', linewidths=0)

    axs0[0].set(xlabel="x (m)", ylabel="x'")
    axs0[0].set_title('Horizontal plane')
    axs0[1].set(xlabel="y (m)", ylabel="y'")
    axs0[1].set_title('Vertical plane')

    for ax in axs0:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0),
                            useMathText=True)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.legend()

    plt.subplots_adjust(
        left=0.12, bottom=0.12, top=0.81, right=0.97, wspace=0.38)
    plt.show()
