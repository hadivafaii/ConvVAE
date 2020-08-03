import os
import torch
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


def plot_latent_scatter(emb, tgt, save_file, num_date_points=5000):
    pd_data = pd.DataFrame({'z0': emb[:num_date_points, 0],
                            'z1': emb[:num_date_points, 1],
                            'lbl': tgt[:num_date_points]})

    sns.set_style('white')
    plt.figure(figsize=(15, 12))
    sns.scatterplot(x='z0', y='z1', hue='lbl', data=pd_data, palette='muted')
    plt.xlabel("z[0]", fontsize=15)
    plt.ylabel("z[1]", fontsize=15)
    plt.savefig(save_file, facecolor='white', dpi=200)
    plt.close()


def plot_latent(decoder, scale=2.0, num_partition=30):
    # display a n*n 2D manifold of digits
    digit_size = 28
    figsize = 15
    figure = np.zeros((digit_size * num_partition, digit_size * num_partition))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, num_partition)
    grid_y = np.linspace(-scale, scale, num_partition)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).cuda()
            x_decoded = decoder(z_sample).squeeze()
            digit = to_np(x_decoded)
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = num_partition * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]", fontsize=15)
    plt.ylabel("z[1]", fontsize=15)
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def convert_time(time_in_secs):

    d = time_in_secs // 86400
    h = (time_in_secs - d * 86400) // 3600
    m = (time_in_secs - d * 86400 - h * 3600) // 60
    s = time_in_secs - d * 86400 - h * 3600 - m * 60

    print("\nd / hh:mm:ss   --->   %d / %d:%d:%d\n" % (d, h, m, s))


def make_animation():
    pass
