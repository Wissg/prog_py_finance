import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def portefeuille_NMC(So, r, K, T, N, sigma, Nmc, k0):
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
    adjustedVolatility = sigma * np.sqrt(1 + (k0 / sigma) * np.sqrt(2 / (delta_t * np.pi)))
    S[0] = So
    A[0] = Bs.Delta(0, S[0], K, T, r, adjustedVolatility)
    V[0] = Bs.BS_CALL(0, S[0], K, T, r, adjustedVolatility)
    B[0] = V[0] - A[0] * S[0] - k0 * abs(A[0]) * S[0]
    Erreur[0] = 0
    P[0] = A[0] * S[0] + B[0]
    P_actu[0] = V[0]
    for k in range(0, Nmc):
        adjustedVolatility = sigma * np.sqrt(1 + (k0 / sigma) * np.sqrt(2 / (delta_t * np.pi)))
        S[0] = So
        A[0] = Bs.Delta(0, S[0], K, T, r, adjustedVolatility)
        V[0] = Bs.BS_CALL(0, S[0], K, T, r, adjustedVolatility)
        B[0] = V[0] - A[0] * S[0] - k0 * abs(A[0]) * S[0]
        Erreur[0] = 0
        P[0] = A[0] * S[0] + B[0]
        P_actu[0] = V[0]
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, adjustedVolatility)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t) - k0 * abs((A[i] - A[i + 1])) * S[i + 1]
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, adjustedVolatility)
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
        PL[k] = P_actu[N - 1] - V[N - 1]
    PL = np.sort(PL)
    # plt.plot(t, V, t, P_actu)
    # plt.show()
    #
    # plt.plot(t, A, label='A')
    # plt.plot(t, B, label='B')
    # plt.xlabel('t')
    # plt.ylabel('A & B')
    # plt.legend()
    # plt.show()

    # Calcul de la VaR
    alpha = 0.1
    Index = floor(alpha * Nmc)
    VaR = PL[Index]
    print("value at risk = ", VaR)
    return (VaR)


So = 100
r = 0.05  # interest rate 5%
K = [80]
T = 5
# N = 100
sigma = 0.25  # volatility 25%
Nmc = 1000
k0 = 0.01

List_N = [(i * 20) for i in range(1, 50)]
ValueAtRisk = [0] * len(List_N)
for j in K:
    for i in range(len(List_N)):
        ValueAtRisk[i] = portefeuille_NMC(So, r, j, T, List_N[i], sigma, Nmc, k0)
    plt.scatter(List_N, ValueAtRisk, label="K = " + str(j))
    max_value = max(ValueAtRisk)
    max_index = ValueAtRisk.index(max_value)
    print("K = ", j, "N = ", List_N[max_index])

plt.title("Value at risk for different trading frequencies")
plt.xlabel("Rebalance number")
plt.ylabel("Value at risk")
plt.legend()
plt.savefig('Graph\Value-at-risk-for-different-trading-frequencies.png', bbox_inches="tight")
plt.show()