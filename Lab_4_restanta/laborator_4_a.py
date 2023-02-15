#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from PIL import Image
from IPython.display import display
from scipy import stats

#Laboratorul 4
fig, axs = plt.subplots(3, 2)

def solve(theta):
    #definim modelul
    with pm.Model() as model:
        n = pm.Poisson('n', mu=20)
        Y = pm.Binomial('Y', n=n, p=theta)

        alpha = pm.Uniform('alpha', lower=0, upper=30)

        t = pm.Exponential('t', lam=pm.math.switch(Y, 1 / (alpha + 1), 1 / alpha))
        obs = pm.math.sum(t[pm.math.where(Y == 0)]) <= 15

        trace = pm.sample(2000, cores=2)
        print(trace['obs'])

solve(theta=0.2)

if __name__ == '__main__':
    solve()