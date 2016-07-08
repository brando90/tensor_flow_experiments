import numpy as np
import matplotlib.pyplot as plt

print 'starting'

x = np.linspace(1,10,num=10)
y = np.exp(-x)

y_err = 0.1 + 0.2*np.sqrt(x)
plt.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
plt.show()
