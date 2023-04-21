import matplotlib.pyplot as plt


def plot_type_2(segments):
    """Plot segments.

    Args:
        segments (np.ndarray): Segments to plot.
    """
    # create figure
    _, ax = plt.subplots()
    # plot image
    ax.imshow(segments)
    # set title
    ax.set_title("segments")
    # show plot
    plt.show()
