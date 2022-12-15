import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pymc3 as pm
import arviz as az
from scipy import stats

#Problema 1
#Generam 500 de date dintr-o combinatie de 3 distributii Gausiene
#din fisierul exemplu.py
clusters = 2
n_cluster = [200, 400]
n_total = sum(n_cluster)
means = [5, 0]
std_devs = [2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix));
plt.show();


# Problema 2
# Calibam pe aceasta multime de date un model cu o distributie gaussiana cu 0, 2, 3 sau 4 componente
clusters = [2, 3, 4]
models = []
idatas = []
data = []
for cluster in clusters:
    with pm.Model() as model:
        μ = pm.Uniform('μ', lower=20, upper=90)

        σ = pm.HalfNormal('σ', sd=10)

        y = pm.Normal('y', mu=μ, sd=σ, observed=data)

        #idata , S22
        idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)

        idatas.append(idata)
        models.append(model)

y_pred_g = pm.sample_posterior_predictive(idata, model=model, keep_size=True)

az.concat(idata, az.from_dict(posterior_predictive=y_pred_g), inplace=True)

ax = az.plot_ppc(idata, num_pp_samples=200, figsize=(32, 18), mean=False)

#Problema 3
#Comparam cele 3 modele folosind WAIC
#comparatie din curs
comp = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="waic", scale="deviance")
az.plot.compare(comp)