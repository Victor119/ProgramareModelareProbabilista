#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from PIL import Image
from IPython.display import display
from scipy import stats

#Laboratorul12
def main():
    def post(theta, Y, alpha=1, beta=1):
        if 0 <= theta <= 1:
            prior = stats.beta(alpha, beta).pdf(theta)
            like = stats.bernoulli(theta).pmf(Y).prod()
            prob = like * prior
        else:
            prob = -np.inf
        return prob

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Ciclam prin valorile dorite pentru can_sd
    for idx, can_sd in enumerate([0.2, 1]):
        Y = stats.bernoulli(0.7).rvs(20)
        n_iters = 1000
        alpha = beta = 1
        theta = 0.5
        trace = {"theta": np.zeros(n_iters)}
        p2 = post(theta, Y, alpha, beta)
        for iter in range(n_iters):
            theta_can = stats.norm(theta, can_sd).rvs(1)
            p1 = post(theta_can, Y, alpha, beta)
            pa = p1 / p2
            if pa > stats.uniform(0, 1).rvs(1):
                theta = theta_can
                p2 = p1
            trace["theta"][iter] = theta

        az.plot_trace(trace, kind='trace', var_names=['theta'], axes=axs[idx:idx + 1])
        axs[idx, 0].set_title('theta can_sd = {}'.format(can_sd))
        axs[idx, 1].set_title('theta can_sd = {}'.format(can_sd))

        print("Can_sd:", can_sd)
        print(az.summary(trace))
        print("Effective sample size (ESS):", az.ess(trace)["theta"].item())
        print("Autocorrelation (lag=1):", az.autocorr(trace["theta"]).item(1))
        print("\n\n\n")

    plt.show()

if __name__ == '__main__':
    main()