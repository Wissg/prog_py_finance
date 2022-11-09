import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def B(t, T, gamma):
    return (1 - np.exp(-gamma * (T - t))) / gamma


def A(t, T, gamma, etha, sigma):
    return ((B(t, T, gamma) - T + t) * (etha * gamma - sigma ** 2 / 2) / (gamma ** 2)) - (
                sigma ** 2 * B(t, T, gamma) ** 2) / (4 * gamma)


def Vavisek(t, T, r0, gamma, etha, sigma):
    return -np.log(np.exp(A(t, T, gamma, etha, sigma) - r0 * B(t, T, gamma))) / (T - t)


def Yield(t, T, r0, gamma, etha, sigma):
    return (-A(t, T, gamma, etha, sigma) + r0 * B(t, T, gamma)) / (T - t)


t = 0
r0 = 0.027
N = 10000
T = np.linspace(0.001, 30, N + 1)
gamma = 0.25
etha = 0.25 * 0.03
sigma = 0.02
p = np.zeros(N + 1)
for i in range(0, N + 1):
    p[i] = Yield(t, T[i], r0, gamma, etha, sigma)
plt.plot(T, p)
plt.axhline(y=r0, color='r', linestyle='--')
plt.show()
