import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def dirac(j, i):
    if j == i:
        return 1
    else:
        return 0


def sigma_locale(K, i, Betha1, Betha2, h):
    return Betha1 / (K[i] ** Betha2) + h


def Crank_Nicolson(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, sigma):
    K = np.linspace(0, Kmax, N + 2)
    deltaK = Kmax / (N + 1)
    T = np.linspace(0, Tmax, M + 2)
    deltat = Tmax / (M + 1)

    V = np.zeros(shape=(M + 2, N + 2))
    C = np.zeros(shape=(M + 2, N + 2))
    C2 = np.zeros(shape=(M + 2, N + 2))
    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    D = np.zeros(N + 1)
    D2 = np.zeros(N + 1)

    for i in range(N + 2):
        V[0, i] = np.maximum(S0 - K[i], 0)

    for n in range(1, M + 2):
        V[n, 0] = S0
        V[n, N + 1] = 0

    for n in range(0, M + 1):
        for i in range(1, N + 1):
            A[i] = (deltat / 4) * (r * K[i] / deltaK - sigma[i] ** 2 * ((K[i] / deltaK) ** 2))
            B[i] = -(deltat / 4) * (
                    r * K[i] / deltaK + sigma[i] ** 2 * ((K[i] / deltaK) ** 2))
            D[i] = 1 + (deltat / 2) * sigma[i] ** 2 * (K[i] / deltaK) ** 2
            C[n, i] = -B[i] * V[n, i - 1] + (
                    1 - (deltat / 2) * sigma[i] ** 2 * (K[i] / deltaK) ** 2) * V[n, i] - \
                      A[i] * V[n, i + 1] - dirac(1, i) * B[1] * S0
        D2[1] = D[1]
        C2[n, 1] = C[n, 1]
        for i in range(2, N + 1):
            D2[i] = D[i] - (B[i] * A[i - 1]) / D2[i - 1]
            C2[n, i] = C[n, i] - (B[i] * C2[n, i - 1]) / D2[i - 1]
        V[n + 1, N] = C2[n, N] / D2[N]
        for i in range(N - 1, 0, -1):
            V[n + 1, i] = (C2[n, i] - A[i] * V[n + 1, i + 1]) / D2[i]
    return V


def Dupire_Price(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, h, method='CEV'):
    sigma = np.zeros(N + 1)
    K = np.linspace(0, Kmax, N + 2)
    if method == 'CEV':
        for i in range(1, N + 1):
            sigma[i] = sigma_locale(K, i, Betha1, Betha2, h)
    # if method=='GATHERAL':
    #     for i in range(1, N + 1):
    #         sigma[i] = bg*(pg*(K[i]-mg)+np.sqrt((K[i]-mg)**2 + ag**2)) +h
    if isinstance(method, int):
        for i in range(1, N + 1):
            sigma[i] = method + h
    V = Crank_Nicolson(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, sigma)
    return V


def Vega_Dupire(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, h, method='CEV'):
    return (Dupire_Price(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, h, method) - Dupire_Price(Kmax, S0, r, Tmax, N, M,
                                                                                            Betha1, Betha2, 0,
                                                                                            method)) / h


def Prix_Dupire_Utiles(Beta1, Beta2, S0, r, Tmax, Kmax, M, N, h, Kp):
    V = Dupire_Price(Kmax, S0, r, Tmax, N, M, Beta1, Beta2, h)
    p = np.zeros((len(Kp))).astype(int)
    dk = Kmax / (N + 1)
    for i in range(len(Kp)):
        p[i] = Kp[i] / dk

    return V[50, p], p


def Vega_Utiles(Beta1, Beta2, S0, r, Tmax, Kmax, M, N, h):
    Vega = Vega_Dupire(Kmax, S0, r, Tmax, N, M, Beta1, Beta2, h)
    _, p = Prix_Dupire_Utiles(Beta1, Beta2, S0, r, Tmax, Kmax, M, N, h, Kp)
    return Vega[50, p]


def LevenbergMarquard(S0, r, Tmax, Kmax, M, N, epsilon, lamb, Kp, Vp, Beta1, Beta2):
    d = [1, 1]
    k = 0
    res = np.zeros(len(Kp))
    Jacobien = np.zeros((len(Kp), 2))
    while np.linalg.norm(d, 2) > epsilon:
        k = k + 1
        Vdupire, _ = Prix_Dupire_Utiles(Beta1, Beta2, S0, r, Tmax, Kmax, M, N, 0, Kp)
        Vega = Vega_Utiles(Beta1, Beta2, S0, r, Tmax, Kmax, M, N, 0.01)
        for i in range(len(Kp)):
            res[i] = Vp[i] - Vdupire[i]
            Jacobien[i, 0] = -Vega[i] / (Kp[i] ** Beta2)
            Jacobien[i, 1] = Vega[i] * (np.log(Kp[i]) * Beta1) / (Kp[i] ** Beta2)

        Hesienne = (np.dot(Jacobien.T, Jacobien) + lamb * np.identity(2))
        d = -np.dot(np.linalg.inv(np.dot(Jacobien.T, Jacobien) + lamb * np.identity(2)),
                    np.dot(Jacobien.T, res))
        Beta1 = Beta1 + d[0]
        Beta2 = Beta2 + d[1]
    print("k = ", k)
    print("Hesienne = ",Hesienne)
    return Beta1, Beta2


Kmax = 20
S0 = 10
r = 0.1
Tmax = 0.5
N = 199
M = 49
Beta1 = 1
Beta2 = 1
h = 0.1
epsilon = 0.00001
lamb = 0.001
K = np.linspace(0, Kmax, N + 2)
T = np.linspace(0, Tmax, M + 2)
V = Dupire_Price(Kmax, S0, r, Tmax, N, M, Beta1, Beta2, h)

# plt.plot(K, V[0, :])
# plt.xlabel("K")
# plt.ylabel("V")
# plt.title("Price Dupire T = 0")
# plt.show()
#
# plt.plot(K, V[floor((M+1)/2), :])
# plt.xlabel("K")
# plt.ylabel("Vega")
# plt.title("Price Dupire T/2")
# plt.show()
#
# plt.plot(K, V[M+1, :])
# plt.xlabel("K")
# plt.ylabel("Vega")
# plt.title("Price Dupire T")
# plt.show()
#
# V = Dupire_Price(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, h, 0.3)
#
# plt.plot(K, V[0, :])
# plt.xlabel("K")
# plt.ylabel("V")
# plt.title("Price Dupire T = 0 sigmal= 0,3")
# plt.show()
#
# plt.plot(K, V[floor((M+1)/2), :])
# plt.xlabel("K")
# plt.ylabel("Vega")
# plt.title("Price Dupire T/2 sigmal= 0,3")
# plt.show()
#
# plt.plot(K, V[M+1, :])
# plt.xlabel("K")
# plt.ylabel("Vega")
# plt.title("Price Dupire T sigmal= 0,3")
# plt.show()
#
# fig = plt.figure()
# k, t = np.meshgrid(K, T)
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(k, t, V)
# ax.set_title("Price Dupire")
# ax.set_xlabel("K")
# ax.set_ylabel("T")
# ax.set_zlabel("V")
# # plt.legend()
# plt.show()
# Vega = Vega_Dupire(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, h)
#
# plt.plot(K, Vega[M+1, :])
# plt.xlabel("K")
# plt.ylabel("Vega")
# plt.title("Vega Dupire Tmax with sigma loc = 1/K")
# plt.show()
#
#
# fig = plt.figure()
# k, t = np.meshgrid(K, T)
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(k, t, Vega)
# ax.set_title("Vega Dupire with sigma loc = 1/K")
# ax.set_xlabel("K")
# ax.set_ylabel("T")
# ax.set_zlabel("Vega")
# plt.show()
#
# Vega = Vega_Dupire(Kmax, S0, r, Tmax, N, M, Betha1, Betha2, h, 0.3)
# print(Vega)
#
# plt.plot(K, Vega[M+1, :])
# plt.xlabel("K")
# plt.ylabel("Vega")
# plt.title("Vega Dupire Tmax with sigma loc = 0.3")
# plt.show()
#
#
# fig = plt.figure()
# k, t = np.meshgrid(K, T)
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(k, t, Vega)
# ax.set_title("Vega Dupire with sigma loc = 0.3")
# ax.set_xlabel("K")
# ax.set_ylabel("T")
# ax.set_zlabel("Vega")
# plt.show()

Kp = [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14]
Vp = [3.3634, 2.9092, 2.4703, 2.0536, 1.6666, 1.3167, 1.0100, 0.7504, 0.5389, 0.3733, 0.2491, 0.1599, 0.0986, 0.0584,
      0.0332]

Beta1, Beta2 = LevenbergMarquard(S0, r, Tmax, Kmax, M, N, epsilon, lamb, Kp, Vp, Beta1, Beta2)
print("Betha1 = ", Beta1, " Betha2 =", Beta2)
