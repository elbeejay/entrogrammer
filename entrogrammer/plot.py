"""Plotting odds and ends, included for users' convenience."""

import matplotlib.pyplot as plt


def plot_entrogram(win_size, HR, labels=True):
    """Make simple plot of entrogram given window sizes and HR."""
    ax1 = plt.gca()  # make plot on current axis if possible
    # plot line and scatter points
    ax1.plot(win_size, HR)
    ax1.scatter(win_size, HR)
    # label axes and titles if labels==True
    if labels is True:
        ax1.set_title('Entrogram')
        ax1.set_xlabel('Length Scale')
        ax1.set_ylabel(r'$H_R$')
