import numpy as np
import matplotlib.pyplot as plt

def plotSpaced(ax,x,y):
    ymax = y.max()
    yoffset = ymax
    for i in range(y.shape[1]):
        ax.plot(x,y[:,i]+yoffset*i)

    ax.set_yticks([])

def change_violin_colors(violin_parts, color_list):
    for pc, linecolor in zip(violin_parts["bodies"], color_list):
        pc.set_facecolor(linecolor)
        pc.set_alpha(1)


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'black' if w > 0 else 'white'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    