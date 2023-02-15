#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from PIL import Image
from IPython.display import display
from scipy import stats

#Laboratorul 1 + 2
def main():
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    #prima distributie este exponentiala
    dst_lambda = 0.5
    mean, std = 1 / dst_lambda, 1 / dst_lambda

    A_1_x = np.linspace(stats.expon.ppf(0.01, scale=mean), stats.expon.ppf(0.99, scale=mean), 1000)
    A_1_y_theoretical = stats.expon.pdf(A_1_x, scale=mean)
    A_1_y_1000_samples = np.random.exponential(scale=mean, size=1000)

    axs[0].plot(A_1_x, A_1_y_theoretical, label='Theoretical pdf (mean={:.4f}, std={:.4f})'.format(mean, std))
    axs[0].hist(
        A_1_y_1000_samples,
        label='1000 samples (mean={:.4f}, std={:.4f})'.format(np.mean(A_1_y_1000_samples), np.std(A_1_y_1000_samples)),
        density=True,
        bins=80
    )
    axs[0].set_title('Exponential')
    axs[0].axes.get_yaxis().set_visible(False)
    axs[0].legend()

    #a doua distributie este normala
    mean, std = 100, 50
    A_2_x = np.linspace(stats.norm.ppf(0.01, mean, std), stats.expon.ppf(0.99, mean, std), 1000)
    A_2_y_theoretical = stats.norm.pdf(A_2_x, mean, std)
    A_2_y_1000_samples = np.random.normal(mean, std, size=1000)

    axs[1].plot(A_2_x, A_2_y_theoretical, label='Theoretical pdf (mean={:.4f}, std={:.4f})'.format(mean, std))
    axs[1].hist(
        A_2_y_1000_samples,
        label='1000 samples (mean={:.4f}, std={:.4f})'.format(np.mean(A_2_y_1000_samples), np.std(A_2_y_1000_samples)),
        density=True,
        bins=80
    )
    axs[1].set_title('Normal')
    axs[1].axes.get_yaxis().set_visible(False)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()