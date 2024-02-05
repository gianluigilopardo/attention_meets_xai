import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib import transforms

import matplotlib.pyplot as plt

import numpy as np

matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)
plt.rcParams.update({'font.size': 18})


def make_box(x, y, ls, lc, alphas, save=None, **kw):
    plt.figure(figsize=(10, 0.8))
    t = plt.gca().transData
    fig = plt.gcf()
    plt.axis('off')
    plt.ylim(0, 0.1)

    for s, c, alpha in zip(ls, lc, alphas):
        text = plt.text(x, y, s, transform=t, **kw)
        text.set_bbox(dict(facecolor=c, alpha=alpha, edgecolor=c))
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width + 15, units='dots')
    if save:
        plt.tight_layout()
        plt.savefig(save, bbox_inches='tight', transparent=True, pad_inches=0)


def plot_exp(exp, tokens, save=None):
    weights = np.array(list(exp.values()))
    scale = max(np.abs(weights))

    colors = []
    alphas = []

    for token in tokens:
        weight = exp[token]
        scaled_weight = weight / scale

        if scaled_weight >= 0:
            colors.append("green")
            alphas.append(scaled_weight)
        else:
            colors.append("red")
            alphas.append(abs(scaled_weight))
    make_box(0.05, 0.1, tokens, colors, alphas, size=20, save=save)
