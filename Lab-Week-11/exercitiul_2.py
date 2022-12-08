import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from scipy import stats

#generam datele

#exercitiul 2 - facem pentru 500 de date

az.style.use('arviz-darkgrid')
dummy_data = np.random.random((500,2))
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



# 1) a) sd = 100 facem reprezentarea grafica a celor 2 modele
# model liniar
with pm.Model() as model_l:
    alpha = pm.Normal('α', mu=0, sd=100)
    beta = pm.Normal('β', mu=0, sd=10)
    eps = pm.HalfNormal('ε', 5)
    niu = alpha + beta * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=niu, sd=eps, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

# modelul cubic
with pm.Model() as model_p:
    alpha = pm.Normal('α', mu=0, sd=100)
    beta = pm.Normal('β', mu=0, sd=10, shape=order)
    eps = pm.HalfNormal('ε', 5)
    niu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=niu, sd=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)


#reprezentarea grafica a celor 2 modele
x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
y_l_post = α_l_post + β_l_post * x_new

plt.plot(x_new, y_l_post, 'C1', label='linear model')

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()


#1) b) sd=np.array([10, 0.1, 0.1, 0.1, 0.1]) - realizam reprezentarea grafica a celor 2 modele
with pm.Model() as model_l:
    alpha = pm.Normal('α', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]))
    beta = pm.Normal('β', mu=0, sd=10)
    eps = pm.HalfNormal('ε', 5)
    niu = alpha + beta * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=niu, sd=eps, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

#modelul cubic
with pm.Model() as model_p:
    alpha = pm.Normal('α', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]))
    beta = pm.Normal('β', mu=0, sd=10, shape=order)
    eps = pm.HalfNormal('ε', 5)
    niu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=niu, sd=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)


#reprezentarea grafica a modelelor
x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
y_l_post = α_l_post + β_l_post * x_new

plt.plot(x_new, y_l_post, 'C1', label='linear model')

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()