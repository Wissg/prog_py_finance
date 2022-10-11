import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs



def Find_volatility_implicite(t, S, K, T, r, M, epsilon):
    sigma = []
    vol = []

    for i in range(0, len(K)):
        if M[i] < S and M[i] > max(S - K[i] * np.exp(-r * T), 0):
            sigma.append(np.sqrt(2 * np.abs(np.log(S / K[i]) + r * T) / T))
            while np.abs(Bs.F_Call(M[i], t, S, K[i], T, r, sigma[len(sigma) - 1])) > epsilon:
                sigma.append(
                    sigma[len(sigma) - 1] - Bs.F_Call(M[i], t, S, K[i], T, r, sigma[len(sigma) - 1]) / Bs.Vega(t, S, K[i], T,
                                                                                                               r, sigma[
                                                                                                           len(
                                                                                                               sigma) - 1]))
            vol.append(sigma[len(sigma) - 1])
            sigma = []
        else:
            "condition d'absence d'arbitrage"
            vol.append(0)
            sigma = []
    return vol


def condition_arbitrage(Vol, K):
    ii = np.where(np.array(Vol) == 0)[0]
    ii = np.array(ii)
    Vol1 = []
    K1 = []
    z = 0
    for i in range(0, len(Vol)):
        "suppressions des sigmas == 0"
        for j in range(0, len(ii)):
            if i == ii[j]:
                z = 1
        if z == 0:
            Vol1.append(Vol[i])
            K1.append(K[i])
        z = 0
    return Vol1, K1


def condition_arbitrage_3d(Vol, K, T):
    ii = np.where(np.array(Vol) == 0)[0]
    ii = np.array(ii)
    Vol1 = []
    K1 = []
    Y1 = []
    z = 0
    for i in range(0, len(Vol)):
        "suppressions des sigmas == 0"
        for j in range(0, len(ii)):
            if i == ii[j]:
                z = 1
        if z == 0:
            Vol1.append(Vol[i])
            K1.append(K[i])
            Y1.append(T[i])

        z = 0
    return Vol1, K1, Y1


K = [5125, 5225, 5325, 5425, 5525, 5625, 5725, 5825]
M = [475, 405, 340, 280.5, 226, 179.5, 139, 105]
So = 5430.3
T = 4 / 12
r = 0.05
t = 0
epsilon = 0.0001
Bs.test()
Vol = Find_volatility_implicite(t, So, K, T, r, M, epsilon)
Vol, K = condition_arbitrage(Vol, K)
plt.plot(K, Vol, label='Vol implicite sans zero')
plt.xlabel('K')
plt.ylabel('Sigma')
plt.legend()
plt.show()

Y = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1, 9):
    Y.append(i / 12)

x, a = np.meshgrid(K, Y)

Vola = []

for i in range(len(Y)):
    B = Find_volatility_implicite(t, So, K, Y[i], r, M, epsilon)
    B,K,Y= condition_arbitrage_3d(B,K,Y)
    Vola.append(B)
    ax.scatter(K, Y[i], B)

ax.set_title("Volatilite implicite")
ax.set_xlabel("K")
ax.set_ylabel("T")
ax.set_zlabel("sigma")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, a, np.array(Vola))

ax.set_title("Volatilite implicite")
ax.set_xlabel("K")
ax.set_ylabel("T")
ax.set_zlabel("sigma")
plt.show()
