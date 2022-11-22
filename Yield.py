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


def B_derivative(gamma, t, T):
    return ((T - t) * np.exp(-gamma * (T - t)) - B(t, T, gamma)) / gamma


def A_derivative(gamma, sigma, etha, t, T):
    return (etha * (B_derivative(gamma, t, T) * gamma - B(t, T, gamma)) + (T - t) * etha - 0.5 * sigma ** 2 * (
            B_derivative(gamma, t, T) - 2 * B(t, T, gamma) / gamma) - (
                    (T - t) / gamma) * sigma ** 2 - 0.25 * sigma ** 2 * B(t, T, gamma) * (
                    2 * gamma * B_derivative(gamma, t, T) - B(t, T, gamma))) / gamma ** 2


def Vavisek(t, T, r0, gamma, etha, sigma):
    return -np.log(np.exp(A(t, T, gamma, etha, sigma) - r0 * B(t, T, gamma))) / (T - t)


def Yield(t, T, r0, gamma, etha, sigma):
    return (-A(t, T, gamma, etha, sigma) + r0 * B(t, T, gamma)) / (T - t)


def Calibration_yield(Ym, T, epsilon, t, r, lamb, etha, gamma, sigma):
    Res = [0] * 10
    Yth = [0] * 10
    Jacobien = np.zeros(shape=(10, 3))
    d = [1,1,1]
    while np.linalg.norm(d, 2) > epsilon:
        for p in range(len(T)):
            tau = T[p] - t
            Yth[p] = Yield(t, T[p], r, gamma, etha, np.sqrt(sigma))
            Res[p] = Ym[p] - Yth[p]
            Jacobien[p, 0] = (B(t, T[p], gamma) - tau) / (gamma * tau)
            Jacobien[p, 1] = -1 * ((1 / (tau * gamma)) * (
                    (B(t, T[p], gamma) - tau) / (2 * gamma) + ((B(t, T[p], gamma) ** 2) / 4)))
            Jacobien[p, 2] = 1 / tau * (
                        A_derivative(gamma, np.sqrt(sigma), etha, t, T[p]) - r * B_derivative(gamma, t, T[p]))

        d = -np.dot(np.linalg.inv(np.dot(Jacobien.T, Jacobien) + lamb * np.identity(3)),
                    np.dot(Jacobien.T, Res))
        etha = etha + d[0]
        sigma = sigma + d[1]
        gamma = gamma + d[2]

    plt.plot(T, Ym, label='Market Yields')
    plt.plot(T, Yth, label='Vasicek Yields')
    plt.legend()
    plt.show()
    return 0


t = 0
r0 = 0.023
N = 10000
T = np.linspace(0.001, 30, N + 1)
gamma = 0.25
etha = 0.25 * 0.03
sigma = 0.02
lamb = 0.01
epsilon = 10 ** (-9)
Y_m = [0.056, 0.064, 0.074, 0.081, 0.082, 0.09, 0.087, 0.092, 0.0895, 0.091]
Ym = [0.035, 0.041, 0.0439, 0.046, 0.0484, 0.0494, 0.0507, 0.0517, 0.052, 0.0523]
T = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
# p = np.zeros(N + 1)
# for i in range(0, N + 1):
#     p[i] = Yield(t, T[i], r0, gamma, etha, sigma)
# plt.plot(T, p)
# plt.axhline(y=r0, color='r', linestyle='--')
# plt.show()

Calibration_yield(Ym, T, epsilon, t, r0, lamb, etha, gamma, sigma)
