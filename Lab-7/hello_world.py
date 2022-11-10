import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


ans = pd.read_csv(r'Prices.csv')


x_1 = ans['Speed'].values
x_2 = np.log(ans['HardDrive'].values)
y = ans['Price'].values
X = np.vstack((x_1, x_2)).T

alpha_real = 60
beta_real_1 = 20
beta_real_2 = 0.5
y_real = alpha_real + beta_real_1 * x_1 + beta_real_2 * x_2
_, ax = plt.subplots(1, 3)

ax[0].plot(x_1, y, 'C0.')
ax[0].set_xlabel('x_1')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x_1, y_real, 'k')

ax[1].plot(x_2, y, 'C1.')
ax[1].set_xlabel('x_2')
ax[1].set_ylabel('y', rotation=0)
ax[1].plot(x_2, y_real, 'k')

az.plot_kde(y, ax=ax[2])

plt.tight_layout()
plt.show()

with pm.Model() as model_poly:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta_1 = pm.Normal('beta_1', mu=0, sd=5)
    beta_2 = pm.Normal("beta_2", mu=0, sd=5)
    eps = pm.HalfCauchy('eps', sd=5)
    mu = pm.Deterministic('mu', alpha + beta_1 * x_1 + beta_2 * x_2)
    y_pred = pm.Normal('y_pred', mu=mu, sd=eps, observed=y)
    idata_g = pm.sample(2000, return_inferencedata=True)

az.plot_trace(idata_g, var_names=['alpha', 'beta_1', 'beta_2', 'eps'])

