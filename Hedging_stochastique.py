import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def sig():
    if np.random.binomial(1, 0.9) == 1:
        return 0.3
    else:
        return 0.5


def sigmasaut(a):
    if a == 0.5:
        if np.random.binomial(1, 0.05) == 1:
            return 0.3
        else:
            return 0.5
    if a == 0.3:
        if np.random.binomial(1, 0.05) == 1:
            return 0.5
        else:
            return 0.3


def portefeuille_volatilite_implicite(So, r, K, T, N, Nmc, sigma):
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
    PL = np.zeros(Nmc)
    S[0] = So
    A[0] = Bs.Delta(0, S[0], K, T, r, sigma[0])
    V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma[0])
    B[0] = 1
    Erreur[0] = 0
    P[0] = A[0] * S[0] + B[0]
    P_actu[0] = V[0]
    W[0] = (A[0] * S[0]) / P_actu[0]
    for k in range(0, Nmc):
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigma[i + 1] ** 2 / 2) * delta_t + sigma[i + 1] * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i] * S[i + 1] + B[i + 1] * (1 + r * delta_t)
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
            W[i + 1] = (A[i + 1] * S[i + 1]) / P_actu[i + 1]
        PL[k] = V[N - 1] - P_actu[N - 1]
        PL[k] = P_actu[N - 1] - np.max(S[N - 1] - K, 0)
    PL = np.sort(PL)

    plt.plot(t, sigma)
    plt.xlabel("Temps")
    plt.ylabel("sigma")
    plt.show()

    return (PL)


So = 1
r = 0.05
K = 1.5
T = 5
N = 100
sigma = np.zeros(N + 1)
Nmc = 1000
for i in range(0, N + 1):
    sigma[i] = sig()

ProfitandLoss1 = portefeuille_volatilite_implicite(So, r, K, T, N, Nmc, sigma)
sigma[0] = 0.3
sns.kdeplot(ProfitandLoss1)
plt.show()

for i in range(1, N + 1):
    sigma[i] = sigmasaut(sigma[i - 1])

ProfitandLoss = portefeuille_volatilite_implicite(So, r, K, T, N, Nmc, sigma)

sns.kdeplot(ProfitandLoss1)
plt.show()
