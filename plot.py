import random

import matplotlib.pylab as plt
import numpy as np



def plot(x_label, y_label, ys, legend=None, title=None, figure_size=(8.0, 4.0), save_path=None):
    """plot multi curve"""
    plt.figure(1, figsize=figure_size)
    if title:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(len(ys)):
        plt.plot(range(len(ys[i])), ys[i])
    if legend:
        plt.legend(legend)
    if save_path:
        plt.savefig(save_path + "/figure.svg", format="svg")
    plt.show()


def get_MA(avr_T, data):
    res = []
    for i in range(len(data) - avr_T + 1):
        res.append(np.mean(data[i:i + avr_T]))
    return np.array(res)


def AVplot(ys, axes=None, shadow=None, label=None, color=None, ratio=1, n=None, line_width=1.5, alpha=0.3):
    """plot average curves"""
    ys = np.array(ys)
    average = np.average(ys, 0)
    if n:
        xs = np.arange(len(average)) / len(average) * n
    else:
        xs = range(len(average))
    if axes:
        axes.plot(xs, average, linewidth=line_width, label=label, color=color)
    else:
        plt.plot(xs, average, linewidth=line_width, label=label, color=color)

    if shadow:
        if shadow == "range":
            upper = np.max(ys, 0)
            lower = np.min(ys, 0)
        else:
            std = np.std(ys, 0)
            upper = average + std * ratio
            lower = average - std * ratio
        if color:
            if axes:
                axes.fill_between(xs, upper, lower, alpha=alpha, color=color)
            else:
                plt.fill_between(xs, upper, lower, alpha=alpha, color=color)
        else:
            if axes:
                axes.fill_between(xs, upper, lower, alpha=alpha)
            else:
                plt.fill_between(xs, upper, lower, alpha=alpha)
