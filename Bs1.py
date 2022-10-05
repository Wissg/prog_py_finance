import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs


def F(Marche, t, S, K, T, r, sigma):
    return Bs.BS_CALL(t, So, K, T, r, sigma) - Marche


def Find_volatility_implicite(t, So, K, T, r, M, epsilon):
    sigma = []
    vol = []

    for i in range(0, len(K)):
        if M[i] < So and M[i] > max(So - K[i] * np.exp(-r * T), 0):
            sigma.append(np.sqrt(2 * np.abs(np.log(So / K[i]) + r * T) / T))
            while np.abs(F(M[i], t, So, K[i], T, r, sigma[len(sigma) - 1])) > epsilon:
                sigma.append(
                    sigma[len(sigma) - 1] - F(M[i], t, So, K[i], T, r, sigma[len(sigma) - 1]) / Bs.Vega(t, So, K[i], T,
                                                                                                        r, sigma[
                                                                                                            len(sigma) - 1]))
            vol.append(sigma[len(sigma) - 1])
            sigma = []
        else:
            "condition d'absence d'arbitrage"
            vol.append(0)
            sigma = []
    return vol


def condition_arbitrage(Vol, K, M):
    ii = np.where(np.array(Vol) == 0)[0]
    for i in range(0, len(ii)):
        "suppressions des sigmas == 0"
        Vol.pop(i)
        K.pop(i)
        M.pop(i)

    plt.plot(K, Vol, label='Vol implicite sans zero')
    plt.xlabel('K')
    plt.ylabel('Sigma')
    plt.legend()
    plt.show()


K = [5125, 5225, 5325, 5425, 5525, 5625, 5725, 5825]
M = [475, 405, 340, 280.5, 226, 179.5, 139, 105]
So = 5430.3
T = 4 / 12
r = 0.05
t = 0
epsilon = 0.0001
Vol = Find_volatility_implicite(t, So, K, T, r, M, epsilon)
condition_arbitrage(Vol, K, M)
Y = []
Vola=[]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1, 13):
    Y.append(i / 12)

for i in range(len(Y)):
    Vola.append(Find_volatility_implicite(t,So, K, Y[i], r,M,epsilon))
    ax.scatter(K, Y[i], Vola[i])

ax.set_title("Volatilite implicite")
ax.set_xlabel("K")
ax.set_ylabel("T")
ax.set_zlabel("sigma")
plt.show()
