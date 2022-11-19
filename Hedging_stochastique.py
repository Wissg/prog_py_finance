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
    if np.random.binomial(1, 0.8) == 1:
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

def repartition(Nx, Nmc, PL):
    F = np.zeros(Nx)
    for i in range(0, Nx):
        compteur = 0
        for n in range(0, Nmc):
            if PL[n] < -0.3 + (0.6 / Nx) * (i - 1):
                compteur = compteur + 1
        F[i] = compteur / Nmc
    return (F)


def portefeuille_volatilite_implicite_sig(So, r, K, T, N, Nmc, sigma):
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
    espPL = 0
    for k in range(0, Nmc):
        A[0] = Bs.Delta(0, S[0], K, T, r, sigma[k])
        V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma[k])
        P[0] = A[0] * S[0] + B[0]
        P_actu[0] = V[0]
        W[0] = (A[0] * S[0]) / P_actu[0]
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigma[i + 1] ** 2 / 2) * delta_t + sigma[i + 1] * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i] * S[i + 1] + B[i + 1] * (1 + r * delta_t)
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
            W[i + 1] = (A[i + 1] * S[i + 1]) / P_actu[i + 1]
        PL[k] = P_actu[N - 1] - V[N - 1]
        # PL[k] = P_actu[N - 1] - np.max(S[N - 1] - K, 0)
    #     plt.plot(t, S)
    #     plt.xlabel("Temps")
    #     plt.ylabel("Valeur de l'actif")
    #     plt.title("Nmc chemins du prix de l'actif model 1")
    # plt.savefig('Graph\multi_path_model_1.png')
    # plt.show()
    plt.plot(t, V, label="V")
    plt.plot(t, P_actu, label="P Actulise")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Evolution de la valeur de l'option et du portefeuille de replication")
    plt.show()
    PL = np.sort(PL)

    plt.plot(sigma)
    plt.xlabel("Temps")
    plt.ylabel("sigma")
    plt.show()

    return (PL)

def portefeuille_volatilite_implicite_sigsaut(So, r, K, T, N, Nmc, sigma):
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
    for k in range(0, Nmc):
        A[0] = Bs.Delta(0, S[0], K, T, r, sigma[k])
        V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma[k])
        P[0] = A[0] * S[0] + B[0]
        P_actu[0] = V[0]
        W[0] = (A[0] * S[0]) / P_actu[0]
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigma[i + 1] ** 2 / 2) * delta_t + sigma[i + 1] * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i] * S[i + 1] + B[i + 1] * (1 + r * delta_t)
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
            W[i + 1] = (A[i + 1] * S[i + 1]) / P_actu[i + 1]
        PL[k] = P_actu[N - 1] - V[N - 1]
        # PL[k] = P_actu[N - 1] - np.max(S[N - 1] - K, 0)
    #     plt.plot(t, S)
    #     plt.xlabel("Temps")
    #     plt.ylabel("Valeur de l'actif")
    #     plt.title("Nmc chemins du prix de l'actif model 2")
    # plt.savefig('Graph\multi_path_model_2.png')
    # plt.show()

    plt.plot(t, V, label="V")
    plt.plot(t, P_actu, label="P Actulise")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Evolution de la valeur de l'option et du portefeuille de replication")
    plt.show()

    PL = np.sort(PL)

    plt.plot(sigma)
    plt.xlabel("Temps")
    plt.ylabel("sigma")
    plt.show()

    return (PL)


So = 1
r = 0.05
K = 1.5
T = 5
N = 100
Nmc = 1000
t = np.linspace(-0.3, 0.3, 100)
sigma = np.zeros(Nmc + 1)
sigma[0] = 0.5
for i in range(1, Nmc + 1):
    sigma[i] = sig()

ProfitandLoss1 = portefeuille_volatilite_implicite_sig(So, r, K, T, N, Nmc, sigma)
# sns.kdeplot(ProfitandLoss1)
# plt.xlabel("P&L")
# plt.ylabel("Valeur")
# plt.title("Fonction de densite volatilite implicite")
# plt.savefig('Graph\densite_volatilite_implicite_0-3_0-5.png')
# plt.show()

Repartition_Hedging1 = repartition(100, Nmc, ProfitandLoss1)
# plt.plot(t, Repartition_Hedging1)
# plt.xlabel("P&L")
# plt.ylabel("Valeur")
# plt.title("Fonction de repatition volatilite implicite")
# plt.savefig('Graph\\repatition_volatilite_implicite_0-3_0-5.png')
# plt.show()

for i in range(1, Nmc + 1):
    sigma[i] = sigmasaut(sigma[i - 1])

ProfitandLoss2 = portefeuille_volatilite_implicite_sigsaut(So, r, K, T, N, Nmc, sigma)

# sns.kdeplot(ProfitandLoss2)
# plt.xlabel("P&L")
# plt.ylabel("Valeur")
# plt.title("Fonction de densite volatilite implicite avec transition")
# plt.savefig('Graph\densite_volatilite_implicite_0-3_0-5+transition.png')
# plt.show()

Repartition_Hedging2 = repartition(100, Nmc, ProfitandLoss2)
# plt.plot(t, Repartition_Hedging2)
# plt.xlabel("P&L")
# plt.ylabel("Valeur")
# plt.title("Fonction de repatition volatilite implicite avec transition")
# plt.savefig('Graph\\repatition_volatilite_implicite_0-3_0-5+transition.png')
# plt.show()