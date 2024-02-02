import sys

sys.path.append("../src")

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn import metrics

from src import kernels
from src.conv import conv1d_interpolate

sns.set()

#functions to interpolate

def function1(x):
    return np.sin(2*x)

def function2(x):
    return np.sin(np.power(x, -1))

def function3(x):
    return np.sign(np.sin(8*x))

#static parameters
functions = [function1, function2, function3]
n_samples = 10
n_predictions = 10_000
x = np.linspace(1e-6, 2*np.pi, n_samples)
x_interp = np.linspace(1e-6, 2*np.pi, n_predictions)

for currentFuction in functions:
    y = currentFuction(x)
    yTrue = currentFuction(x_interp)
    #apply each kernel to current fucnction
    yInterpK1 = conv1d_interpolate(x_measure=x, y_measure=y, x_interpolate=x_interp, kernel=kernels.sinc_kernel)
    print(f"MSE: {metrics.mean_squared_error(y_pred=yInterpK1, y_true=yTrue):.8f} for sinc kernel")
    yInterpK2 = conv1d_interpolate(x_measure=x, y_measure=y, x_interpolate=x_interp, kernel=kernels.linear_kernel)
    print(f"MSE: {metrics.mean_squared_error(y_pred=yInterpK2, y_true=yTrue):.8f} for linear kernel")
    yInterpK3 = conv1d_interpolate(x_measure=x, y_measure=y, x_interpolate=x_interp, kernel=kernels.nearest_neighbour_kernel)
    print(f"MSE: {metrics.mean_squared_error(y_pred=yInterpK3, y_true=yTrue):.8f} for nearest neighbour kernel")
    #display results by function
    plt.plot(x, y)
    plt.plot(x_interp, yInterpK1)
    plt.plot(x_interp, yInterpK2)
    plt.plot(x_interp, yInterpK3)
    plt.show()
