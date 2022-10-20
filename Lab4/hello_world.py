import matplotlib.pyplot as plt

import numpy as np

import pymc3 as pm
import arviz as az

import pandas as pd

import random
import math

from scipy import stats


model = pm.Model()

with model:
    media = 1
    deviatiaStandard = 0.5
    #alpha = 1/countData.mean()

    parameter = pm.Exponential("poissonParam", 20)
    dataGenerator = pm.Poisson("dataGenerator", parameter)

    y = pm.Normal('y', mu = media, sd = deviatiaStandard, nu = parameter, observed=dataGenerator)
    idata_t = pm.sample(1000, return_inferencedata=True)



az.plot_trace(idata_t)
