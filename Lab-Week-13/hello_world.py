import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from scipy import stats

# Problema 1
# Alte distributii a priori decat cea din curs folosind metoda grid computing
def posterior_grid(grid_points=1000, heads=3, tails=9):
    #date in cerinta
    grid = np.linspace(0, 0.5, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior = posterior / posterior.sum()
    return (grid, posterior)

data = np.repeat([0, 1], (3, 9))
points = 9
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')


# Problema 2
# Estimam phi cu raportul 4N_int/N
N = 100
x, y = np.random.uniform(-1, 1, size=(2, N))
inside = (x**2 + y**2) <= 1
pi = inside.sum()*4/N
error = abs((pi - np.pi) / pi) * 100
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
#din curs
plt.plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()
