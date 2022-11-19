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
    if np.random.binomial(1, 0.5) == 1:
        return 0.3
    else:
        return 0.5


def portefeuille_historique(So, r, K, T, N, sigma, sigmah, Nmc):
    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    P = np.zeros(N + 1)
    Erreur = np.zeros(N + 1)
    delta_t = T / N
    P_actu = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    t = np.linspace(0, T, N + 1)
    PL = np.zeros(Nmc)
    S[0] = So
    A[0] = Bs.Delta(0, S[0], K, T, r, sigma)
    V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma)
    B[0] = 1
    Erreur[0] = 0
    P[0] = A[0] * S[0] + B[0]
    P_actu[0] = V[0]
    for k in range(0, Nmc):
        A[0] = Bs.Delta(0, S[0], K, T, r, sigma)
        V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma)
        B[0] = 1
        Erreur[0] = 0
        P[0] = A[0] * S[0] + B[0]
        P_actu[0] = V[0]
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigmah ** 2 / 2) * delta_t + sigmah * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
        PL[k] = P_actu[N - 1] - V[N - 1]
    PL = np.sort(PL)


    return (PL)

def portefeuille_volatilite_implicite_sig_historique(So, r, K, T, N, Nmc, sigma, sigmah):
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
    B[0] = 1
    Erreur[0] = 0
    A[0] = Bs.Delta(0, S[0], K, T, r, sigma)
    V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma)
    P[0] = A[0] * S[0] + B[0]
    P_actu[0] = V[0]
    W[0] = (A[0] * S[0]) / P_actu[0]
    for k in range(0, Nmc):
        A[0] = Bs.Delta(0, S[0], K, T, r, sigma)
        V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma)
        P[0] = A[0] * S[0] + B[0]
        P_actu[0] = V[0]
        W[0] = (A[0] * S[0]) / P_actu[0]
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigmah[i + 1] ** 2 / 2) * delta_t + sigmah[i + 1] * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i] * S[i + 1] + B[i + 1] * (1 + r * delta_t)
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
            W[i + 1] = (A[i + 1] * S[i + 1]) / P_actu[i + 1]
        PL[k] = P_actu[N - 1] - V[N - 1]
        # PL[k] = P_actu[N - 1] - np.max(S[N - 1] - K, 0)
    PL = np.sort(PL)

    return (PL)
So = 1
r = 0.05
K = 1.5
T = 5
Nmc = 1000
sigma = 0.5
sigmah = 0.4
N = 100

ProfitandLoss1 = portefeuille_historique(So, r, K, T, N, sigma, sigmah, Nmc)
sns.kdeplot(ProfitandLoss1)
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de densite volatilite implicite= 0,5 et historique= 0,3")
plt.savefig('Graph\portefeuille_historique_0-5_0-4.png')
plt.show()

sigma = 0.4
sigmah = 0.5
ProfitandLoss = portefeuille_historique(So, r, K, T, N, sigma, sigmah, Nmc)
sns.kdeplot(ProfitandLoss)
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de densite volatilite implicite= 0,3 et historique= 0,5")
plt.savefig('Graph\portefeuille_historique_0-4_0-5.png')
plt.show()


sigmah = np.zeros(N + 1)
for i in range(1, N + 1):
    sigmah[i] = sig()

ProfitandLoss3 = portefeuille_volatilite_implicite_sig_historique(So, r, K, T, N, Nmc, sigma, sigmah)
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de implicite= 0,5 et historique suit model 1")
sns.kdeplot(ProfitandLoss3)
plt.savefig('Graph\portefeuille_sig_historique_stoch.png')
plt.show()