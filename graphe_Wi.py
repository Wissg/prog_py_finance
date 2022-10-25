import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def portefeuille(So, r, K, T, N, sigma):
    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    P = np.zeros(N + 1)
    Erreur = np.zeros(N + 1)
    delta_t = T / N
    P_actu = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    t = np.linspace(0, T, N + 1)
    W = np.zeros(N + 1)
    S[0] = So
    A[0] = Bs.Delta(0, S[0], K, T, r, sigma)
    V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma)
    B[0] = 1
    Erreur[0] = 0
    P[0] = A[0] * S[0] + B[0]
    P_actu[0] = V[0]
    W[0] = (A[0] * S[0]) / P_actu[0]
    for i in range(0, N):
        S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
        A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
        B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
        P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
        V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
        P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
        Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
        W[i + 1] = (A[i + 1] * S[i + 1]) / P_actu[i + 1]
    PL = V[N - 1] - P_actu[N - 1]

    plt.plot(t, W)
    plt.xlabel("Temps")
    plt.ylabel("W")
    plt.show()

    return (PL)


So = 1
r = 0.05
K = 1.5
T = 5
N = 100
sigma = 0.5
portefeuille(So, r, K, T, N, sigma)
