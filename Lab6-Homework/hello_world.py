import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pymc3 as pm
import arviz as az

# afisam informatiile din fisierul data.csv
sir_citit = pd.read_csv(r'data.csv')
print(sir_citit)


#Problema 1
#Reprezentam grafic datele pentru rezultatul testului de varsta al mamei
a = sir_citit['momage'].values
b = sir_citit['ppvt'].values

a = a - a.mean()
plt.scatter(a, b)
plt.xlabel('a')
plt.ylabel('b', rotation=0)

#Problema 3
# Determinam care este dreapta de regresie potrivita pentru date
# Varsta recomandata de nastere pentru mama?
alpha_real = 80
beta_real = 1
b_real = alpha_real + beta_real * a
_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(a, b, 'C0.')
ax[0].set_xlabel('a')
ax[0].set_ylabel('b', rotation=0)
ax[0].plot(a, b_real, 'k')
az.plot_kde(b, ax=ax[1])
ax[1].set_xlabel('b')
plt.tight_layout()
plt.show()

#Problema2
#Modelul Bayesian pentru regresie liniara
with pm.Model() as model_poly:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=5)
    beta2 = pm.Normal('beta2', mu=0, sd=5)
    epsilon = pm.HalfCauchy('eps', 5)
    mu = alpha + beta1 * a + beta2 * a**2
    b_pred = pm.Normal('b_pred', mu=mu, sd=epsilon, observed=b)
    idata_g = pm.sample(2000, return_inferencedata=True)
az.plot_trace(idata_g, var_names=['alpha', 'beta1', 'beta2', 'eps'])

#Problema4
# Schimbam varsta cu nivelul de educatie al mamei

a = sir_citit['educ_cat'].values
b = sir_citit['ppvt'].values

plt.scatter(a, b)
plt.xlabel('a')
plt.ylabel('b', rotation=0)

alpha_real = 60
beta_real = 1
y_real = alpha_real + beta_real * a
_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(a, b, 'C0.')
ax[0].set_xlabel('a')
ax[0].set_ylabel('b', rotation=0)
ax[0].plot(a, y_real, 'k')
az.plot_kde(b, ax=ax[1])
ax[1].set_xlabel('b')
plt.tight_layout()
plt.show()

with pm.Model() as model_poly:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=5)
    eps = pm.HalfCauchy('eps', sd=5)
    mu = pm.Deterministic('mu', alpha + beta * a)
    y_pred = pm.Normal('y_pred', mu=mu, sd=eps, observed=b)
    idata_g = pm.sample(3000, return_inferencedata=True)
az.plot_trace(idata_g, var_names=['alpha', 'beta', 'eps'])
