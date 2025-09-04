import random

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import torch


def set_random_seed(seed):
    """
    设置随机种子

    Args:
        seed: 随机种子

    """
    random.seed(seed)
    cp.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot(x_label, y_label, *ys, legend=None, title=None, figure_size=(8.0, 4.0), save_path=None):
    """
    绘制曲线

    Args:
        x_label: x轴标签
        y_label: y轴标签
        ys: 数据
        legend: 图例
        title: 标题
        figure_size: 图大小
        save_path: 保存地址

    """
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

def implot(map, n_bins, feature_ranges, performance_bound, x_label, y_label):
    fitness_list = []
    image = np.zeros((n_bins, n_bins))
    for index, individual in map.items():
        fitness = individual.fitness / 10 - performance_bound[0]
        fitness_list.append(fitness)
        image[index[::-1]] = fitness
    plt.imshow(image, origin="lower")
    y_ticks = np.arange(feature_ranges[1][0], feature_ranges[1][1],
                        (feature_ranges[1][1] - feature_ranges[1][0]) / n_bins) + (
                      feature_ranges[1][1] - feature_ranges[1][0]) / n_bins / 2
    x_ticks = np.arange(feature_ranges[0][0], feature_ranges[0][1],
                        (feature_ranges[0][1] - feature_ranges[0][0]) / n_bins) + (
                      feature_ranges[0][1] - feature_ranges[0][0]) / n_bins / 2
    plt.xticks(np.arange(n_bins), labels=x_ticks.round(2))
    plt.yticks(np.arange(n_bins), labels=y_ticks.round(2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(n_bins):
        for j in range(n_bins):
            plt.text(j, i, str(image[i, j].round(1)), ha="center", va="center", color="w")
    plt.colorbar()
    plt.clim(0, performance_bound[1] - performance_bound[0])
    plt.tight_layout()
    plt.show()
    return fitness_list