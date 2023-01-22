import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 2)

mu = 3
A_1_x = np.arange(stats.poisson.ppf(0.01, mu), stats.poisson.ppf(0.99, mu))
A_1_y = stats.poisson.pmf(A_1_x, mu)

axs[0, 0].plot(A_1_x, A_1_y, label='pdf')
axs[0, 0].set_title('Numărul așteptat de oameni')
axs[0, 0].axes.get_yaxis().set_visible(False)

A_2_x = np.linspace(1.4, 100)
A_2_y = stats.uniform.pdf(A_2_x)

axs[0, 1].plot(A_2_x, A_2_y, label='pdf')
axs[0, 1].set_title('Greutatea câinilor adulţi')
axs[0, 1].axes.get_yaxis().set_visible(False)

A_3_x = np.linspace(1800, 6300)
A_3_y = stats.norm.pdf(A_3_x, 4050, 600)

axs[1, 0].plot(A_3_x, A_3_y, label='pdf')
axs[1, 0].set_title('Greutatea elefanţilor adulţi')
axs[1, 0].axes.get_yaxis().set_visible(False)

A_4_x = np.linspace(40, 120)
A_4_y = stats.skewnorm.pdf(A_4_x, 1.8, 70, 20)

axs[1, 1].plot(A_4_x, A_4_y, label='pdf')
axs[1, 1].set_title('Greutatea oamenilor adulţi')
axs[1, 1].axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
