import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2)

for idx_x, Y_value in enumerate([0, 5, 10]):
    for idx_y, theta in enumerate([0.2, 0.5]):
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)
            Y = pm.Binomial('Y', n=n, p=theta, observed=Y_value)

            trace = pm.sample(1000, cores=1)
            az.plot_posterior(trace, var_names=['n'], ax=axs[idx_x, idx_y])
            axs[idx_x, idx_y].set_title(f'n for Y={Y_value} and θ={theta}')

plt.tight_layout()
plt.show()

"""
b)
Efectul lui Y si theta asupra lui n:

Intrucat Y ~ Binomial(n, theta), pentru a maximiza MAP, centrul distributiei binomiale trebuie sa se aproprie de Y:
 - cu cat theta este mai mic, cu atat centrul distributiei binomiale se duce mai in stanga, deci pentru a muta centrul
   cat mai aproape de Y, media valorilor lui n si valoarea maxima a lui n va trebuie sa creasca
 - cu cat Y este mai mare cu atat si centrul distributiei binomiale creste pentru a fi cat mai aproape de Y, deci atat
   media valorilor lui n si valoarea maxima a lui n vor creste


"""