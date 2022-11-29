import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from scipy import stats

# display data from file
iris = pd.read_csv(r'Admission.csv')
iris.head()
df = iris.query("Admission == (0, 1)")
y_0 = pd.Categorical(df['Admission']).codes
x_n = ['GRE', 'GPA']
x_0 = df[x_n].values
x_c0 = x_0 - x_0[:, 0].mean()
x_c1 = x_0 - x_0[:, 1].mean()
x_c = list(zip(x_c0, x_c1))

# 1) Facem modelul
mom_model = pm.Model()
with mom_model as model_0:
    alpha = pm.Normal('α', mu=0, sd=10)
    beta = pm.Normal('β', mu=0, sd=2, shape=len(x_n))
    miu = alpha + pm.math.dot(x_0, beta)
    teta = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-miu)))
    bd = pm.Deterministic('bd', -alpha / beta[1] - beta[0] / beta[1] * x_0[:, 0])
    yl = pm.Bernoulli('yl', p=teta, observed=y_0)
    idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

# afisam date
idx = np.argsort(x_n[:, 0])
bd = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(x_n[:, 0], x_n[:, 1], c=[f'C{x}' for x in y_0])
plt.plot(x_n[:, 0][idx], bd, color='k')
az.plot_hdi(x_n[:, 0], idata_1.posterior['bd'], color='k')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])

#2) Facem un grafic pentru intervalul de 94% HDI
ppc = pm.sample_posterior_predictive(idata_1, samples=100, model=mom_model)
beta = az.plot_hdi(beta, ppc['beta'], hdi_prob=0.94, color='k')


#3) 550 GRE si 3.5 GPA  90% HDI
x_1 = stats.uniform.logpdf(550, 3.5, loc=0, scale=1)
#   500 GRE si 3.2 GPA, 90% HDI
x_2 = stats.uniform.logpdf(500, 3.2, loc=0, scale=1)
z = x_1 + x_2
az.plot_posterior({'x1': x_1, 'x2': x_2, 'z': z})

y_real = az.plot_hdi(z, ppc['y_pred'], hdi_prob=0.90, color='k')
