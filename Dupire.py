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


def sigma_locale(K, i, Betha1, Betha2):
    return Betha1 / (K[i] ** Betha2)


def Crank_Nicolson(Kmax, S0, r, Tmax, N, M, Betha1, Betha2):
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
            A[i] = (deltat / 4) * (r * K[i] / deltaK - sigma_locale(K, i, Betha1, Betha2) ** 2 * ((K[i] / deltaK) ** 2))
            B[i] = -(deltat / 4) * (r * K[i] / deltaK + sigma_locale(K, i, Betha1, Betha2) ** 2 * ((K[i] / deltaK) ** 2))
            D[i] = 1 + (deltat / 2) * sigma_locale(K, i, Betha1, Betha2)**2 * (K[i] / deltaK) ** 2
            C[n, i] = -B[i] * V[n, i - 1] + (1 - (deltat / 2) * sigma_locale(K, i, Betha1, Betha2)**2 * (K[i] / deltaK) ** 2) * V[n, i] - A[i] * V[n, i + 1] - dirac(1, i) * B[1] * S0
        D2[1] = D[1]
        C2[n, 1] = C[n, 1]
        for i in range(2, N + 1):
            D2[i] = D[i] - (B[i] * A[i - 1]) / D2[i - 1]
            C2[n, i] = C[n, i] - (B[i] * C2[n, i - 1]) / D2[i - 1]
        V[n + 1, N] = C2[n, N] / D2[N]
        for i in range(N - 1, 0, -1):
            V[n + 1, i] = (C2[n, i] - A[i] * V[n + 1, i + 1]) / D2[i]
    plt.plot(K, V[10, :])
    plt.xlabel("K")
    plt.ylabel("V")
    plt.title("Price Dupire")
    # plt.legend()
    # plt.savefig('Graph\Vasicek_interest_rate.png')
    plt.show()

    fig = plt.figure()
    k, t = np.meshgrid(K, T)
    ax = fig.add_subplot( projection='3d')
    ax.plot_surface(k, t, V)
    ax.set_title("Price Dupire")
    ax.set_xlabel("K")
    ax.set_ylabel("T")
    ax.set_zlabel("V")
    plt.show()
    return V


Kmax = 20
S0 = 10
r = 0.1
Tmax = 0.5
N = 199
M = 49
Betha1 = 1
Betha2 = 1
Crank_Nicolson(Kmax, S0, r, Tmax, N, M, Betha1, Betha2)
