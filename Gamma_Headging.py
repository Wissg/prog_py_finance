import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def portefeuille(S0, r, K1, K2, T1, T2, N, sigma):
    A = np.zeros(N)
    B = np.zeros(N)
    P = np.zeros(N)
    Erreur = np.zeros(N)
    delta_t = T1 / N
    P_actu = np.zeros(N)
    GammaV = np.zeros(N)
    GammaC = np.zeros(N)
    V = np.zeros(N)
    G = np.zeros(N)
    S = np.zeros(N)
    t = np.linspace(0, T1, N)

    S[0] = S0
    B[0] = 1
    GammaV[0] = Bs.Gamma(0, S0, K1, T1, r, sigma)
    GammaC[0] = Bs.Gamma(0, S0, K2, T2, r, sigma)
    G[0] = GammaV[0] / GammaC[0]
    A[0] = Bs.Delta(0, S0, K1, T1, r, sigma) - G[0] * Bs.Delta(0, S0, K2, T2, r, sigma)
    P[0] = A[0] * S[0] + B[0] + G[0] * Bs.BS_CALL(0, S0, K2, T2, r, sigma)
    V[0] = Bs.BS_CALL(0, S0, K1, T1, r, sigma)
    P_actu[0] = V[0]
    Erreur[0] = 0

    for i in range(0, N - 1):
        S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
        P[i + 1] = A[i] * S[i + 1] + G[i] * Bs.BS_CALL(t[i + 1], S[i + 1], K2, T2, r, sigma) + B[i] * (1 + r * delta_t)
        G[i + 1] = Bs.Gamma(t[i + 1], S[i + 1], K1, T1, r, sigma) / Bs.Gamma(t[i + 1], S[i + 1], K2, T2, r, sigma)
        A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K1, T1, r, sigma) - G[i + 1] * Bs.Delta(t[i + 1], S[i + 1], K2, T2, r, sigma)
        B[i + 1] = P[i + 1] - A[i + 1] * S[i + 1] - G[i + 1] * Bs.BS_CALL(t[i + 1], S[i + 1], K2, T2, r, sigma)
        V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K1, T1, r, sigma)
        P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
        Erreur[i + 1] = P_actu[i + 1] - V[i + 1]

    plt.plot(t, V, label="V")
    plt.plot(t, P_actu, label="P Actulise")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Evolution de la valeur de l'option et du portefeuille de replication")
    plt.legend()
    plt.savefig('Graph\Gamma_Headging_Evolution_de_la_valeur_de_l_option_et_du_portefeuille_de_replication')
    plt.show()

    plt.plot(t, A, label="A")
    plt.plot(t, B, label="B")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Evolution de la quantité d'option et de la valeur du cash")
    plt.legend()
    plt.savefig('Graph\Gamma_Headging_Evolution_de_la_quantité_d_option_et_de_la_valeur_du_cash')
    plt.show()

    plt.plot(t, Erreur)
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Erreur du portefeuille de replication")
    plt.savefig('Graph\Gamma_Headging_Erreur_du_portefeuille_de_replication')
    plt.show()

    PL = P_actu[N - 1] - V[N - 1]
    return PL


S0 = 1
r = 0.05
K1 = 1.5
K2 = K1
T1 = 5
T2 = 10
sigma = 0.5
N = 100
portefeuille(S0, r, K1, K2, T1, T2, N, sigma)
