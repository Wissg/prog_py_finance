import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs

def portefeuille_NMC(So, r, K, T, N, sigma, Nmc):
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
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
        PL[k] = V[N - 1] - P_actu[N - 1]
    PL = np.sort(PL)

    plt.plot(t, Erreur)
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Erreur du replicating portfolio")
    plt.show()

    return (PL)

def portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc,mod):
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
        for i in range(0, N):
            if (i+1)%mod == 0:
                S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
                A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
                B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
                P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
                V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
                P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
                Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
            else:
                S[i + 1] = S[i]
                A[i + 1] = A[i]
                B[i + 1] = B[i]
                P[i + 1] = P[i]
                V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
                P_actu[i + 1] = P_actu[i]
                Erreur[i + 1] = Erreur[i]
        PL[k] = V[N - 1] - P_actu[N - 1]
    PL = np.sort(PL)
    plt.plot(t, V, t, P_actu)
    plt.show()

    plt.plot(t, A, label='A')
    plt.plot(t, B, label='B')
    plt.xlabel('t')
    plt.ylabel('A & B')
    plt.legend()
    plt.show()
    return (PL)


def repartition(Nx, Nmc, PL):
    F = np.zeros(Nx)
    for i in range(0, Nx):
        compteur = 0
        for n in range(0, Nmc):
            if PL[n] < -0.3 + (0.6 / Nx) * (i - 1):
                compteur = compteur + 1
        F[i] = compteur / Nmc
    return (F)


So = 1
r = 0.05
K = 1.5
T = 5
N = 100
sigma = 0.5
Nmc = 1000
ProfitandLoss = portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc, 2)

Repartition_Hedging = repartition(100, Nmc, ProfitandLoss)

t = np.linspace(-0.3, 0.3, 100)
plt.plot(t, Repartition_Hedging)
plt.show()


sns.histplot(ProfitandLoss, color="red", label="100% Equities", kde=True, stat="density", linewidth=0)
plt.show()

# Calcul de la VaR

alpha = 0.1

Index = floor(alpha * Nmc)
VaR = ProfitandLoss[Index]

stats.probplot(ProfitandLoss, dist="norm", plot=pylab)
pylab.show()
print(VaR)