#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from PIL import Image
from IPython.display import display

#Generam valori x
def generam_valori_x():
	value = np.linspace(-20,10,100)
	return value

#Generam valori y folosind formula: y = 9 + 3*x + N(0,5)
#extragem esantion random dintr-o distributie Gaussiana
def generam_valori_y(x):
	value = 9 + 3 * x + np.random.normal(0, 5, 100)
	return value

def main():
	#Generam valori pentru x
	x = generam_valori_x()

	#Generam valori pentru y
	y = generam_valori_y(x)


	#Generam modelul de forma: (alpha) + (beta)*x + (epsilon)
	#pentru alpha si beta vom folosi o distributie normala
	#pentru epsilon vom folosi distributie HalfStudentT

	basic_model = pm.Model()

	with basic_model:
		alpha = pm.Normal('alpha', mu=0, sigma=1)
		beta = pm.Normal('beta', mu=0, sigma=1)
		sigma_2 = pm.HalfStudentT('sigma', nu=3, sigma=2)
		mu_2 = alpha + beta * x
		step = pm.NUTS()
		likelihood = pm.Normal('y', mu=mu_2, sigma=sigma_2, observed=y)


	with basic_model:
		# deseneam 1000 de posterior samples
		#facem return_inferencedata=False ca sa dam silence la warning
		trace = pm.sample(1000, tune=1000, step=step, return_inferencedata=False)

		#ca sa afisam rezultatele si sa recuperam datele facem urmatorul display
		#pentru a afisa in consola tabela cu doar 2 zecimale
		display(az.summary(trace, round_to=2))

		#verificam rezultatele folosind plot_posterior
		az.plot_posterior(trace)

	plt.show()

if __name__ == '__main__':
	main()

