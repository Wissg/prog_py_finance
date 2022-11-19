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
    espPL = 0
    espCarrePL = 0
    espS = 0
    espCarreS = 0
    for k in range(0, Nmc):
        A[0] = Bs.Delta(0, S[0], K, T, r, sigma)
        V[0] = Bs.BS_CALL(0, S[0], K, T, r, sigma)
        P[0] = A[0] * S[0] + B[0]
        P_actu[0] = V[0]
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
            A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
            Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
        PL[k] = V[N - 1] - P_actu[N - 1]

        espPL = PL[k] + espPL
        espCarrePL = PL[k] ** 2 + espCarrePL
        espS = S[N - 1] + espS
        espCarreS = S[N - 1] ** 2 + espCarreS

        plt.plot(t, S)
        plt.xlabel("Temps")
        plt.ylabel("Valeur de l'actif")
        plt.title("Nmc chemins du prix de l'actif")
    plt.savefig('Graph\multi_path.png')
    plt.show()

    PL = np.sort(PL)

    EsperanceTheorique = S[0] * np.exp((r - 0.5 * sigma ** 2) * T) * np.exp(0.5 * T * sigma ** 2)
    VarianceTheorique = S[0] * np.exp((r - 0.5 * sigma ** 2) * 2 * T) * (np.exp(T * sigma ** 2) - 1) * np.exp(
        T * sigma ** 2)
    espPL = espPL / Nmc
    varPL = espCarrePL / Nmc - espPL ** 2
    espS = espS / Nmc
    varS = espCarreS / Nmc - espS ** 2

    print("Esperance de PL = ", espPL)
    print("Variance de PL = ", varPL)
    # fontion python calcul variance et esperance
    print("Esperance de PL = ", np.mean(PL))
    print("Variance de PL = ", np.std(PL)**2)
    print("Esperance de St = ", espS)
    print("Variance de St = ", varS)
    print("Esperance theorique de St = ", EsperanceTheorique)
    print("Variance theorique de St = ", VarianceTheorique)

    plt.plot(t, Erreur)
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Erreur du replicating portfolio")
    plt.show()

    return (PL)


def portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc, mod):
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
            if (i + 1) % mod == 0:
                S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
                A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
                B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
                P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
                V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
                P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
                Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
            else:
                S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn(1))
                A[i + 1] = A[i]
                B[i + 1] = B[i] * (1 + r * delta_t)
                P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
                V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
                P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])
                Erreur[i + 1] = P_actu[i + 1] - V[i + 1]
        PL[k] = V[N - 1] - P_actu[N - 1]
    PL = np.sort(PL)
    # plt.plot(t, V, t, P_actu)
    # plt.show()
    #
    plt.plot(t, A, label='A')
    plt.plot(t, B, label='B')
    plt.xlabel('t')
    plt.ylabel('Valeur')
    plt.title("ratio de la couverture N trading = 10")
    plt.legend()
    plt.savefig('Graph\\ratio_de_la_couverture')
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
t = np.linspace(-0.3, 0.3, 100)

ProfitandLoss = portefeuille_NMC(So, r, K, T, N, sigma, Nmc)
Repartition_Hedging = repartition(100, Nmc, ProfitandLoss)
plt.plot(t, Repartition_Hedging, label='hedging chaque dt')
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de repartition")
plt.savefig('Graph\integral_chaque_dt.png')
plt.show()
sns.kdeplot(ProfitandLoss, label='hedging chaque dt')
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de densite")
plt.savefig('Graph\density_chaque_dt.png')
plt.show()

PL10 =portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc, 10)
# Calcul de la VaR
alpha = 0.1
Index = floor(alpha * Nmc)
VaR = PL10[Index]
print("VaR lorsque rebalance une fois sur 10 = ", VaR)

Repartition_Hedging10 = repartition(100, Nmc, PL10)
plt.plot(t, Repartition_Hedging10, label='1 fois sur 10')
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de repartition rebalance 1 fois sur 10")
plt.axhline(y=0.1, color='r', linestyle='--', label="alpha = 0.1")
plt.legend()
plt.savefig('Graph\integral_1_fois_sur_10+VaR.png')
plt.show()

ProfitandLoss = portefeuille_NMC(So, r, K, T, N, sigma, Nmc)
# Calcul de la VaR
alpha = 0.1
Index = floor(alpha * Nmc)
VaR = ProfitandLoss[Index]

print("VaR lorsque rebalance chaque dt = ", VaR)

Repartition_Hedging = repartition(100, Nmc, ProfitandLoss)
plt.plot(t, Repartition_Hedging, label='chaque dt')
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de repartition rebalance chaque dt")
plt.axhline(y=0.1, color='r', linestyle='--', label="alpha = 0.1")
plt.legend()
plt.savefig('Graph\integral_chaque_dt+VaR.png')
plt.show()

listN = [100, 50, 25, 20, 5, 2, 1]
ProfitandLossfreq = [0] * len(listN)
VarPL = [0] * len(listN)
EspPL = [0] * len(listN)
for j in range(len(listN)):
    ProfitandLossfreq[j] = portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc, N / listN[j])
    VarPL[j] = np.std(ProfitandLossfreq[j]) ** 2
    EspPL[j] = np.mean(ProfitandLossfreq[j])

plt.plot(listN, EspPL)
plt.xlabel("N trading")
plt.ylabel("Moyenne")
plt.title("Moyenne P&L en fonction du nombre de rebalancement")
plt.savefig('Graph\Moyenne_P&L_en_fonction_du_nombre_de_rebalancement.png')
plt.show()
plt.plot(listN, VarPL)
plt.xlabel("N trading")
plt.ylabel("Variance")
plt.title("Variance P&L en fonction du nombre de rebalancement")
plt.savefig('Graph\Variance_P&L_en_fonction_du_nombre_de_rebalancement.png')
plt.show()

ProfitandLoss1 = portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc, 2)
ProfitandLoss2 = portefeuille_NMC_Freq(So, r, K, T, N, sigma, Nmc, 4)

Repartition_Hedging = repartition(100, Nmc, ProfitandLoss)
Repartition_Hedging1 = repartition(100, Nmc, ProfitandLoss1)
Repartition_Hedging2 = repartition(100, Nmc, ProfitandLoss2)
plt.plot(t, Repartition_Hedging, label='hedging chaque dt')
plt.plot(t, Repartition_Hedging1, label='1 fois sur 2')
plt.plot(t, Repartition_Hedging2, label='1 fois sur 4')
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de repartiton")
plt.legend()
plt.savefig('Graph\density.png')
plt.show()

sns.kdeplot(ProfitandLoss, label='hedging chaque dt')
sns.kdeplot(ProfitandLoss1, label='1 fois sur 2')
sns.kdeplot(ProfitandLoss2, label='1 fois sur 4')
plt.xlabel("P&L")
plt.ylabel("Valeur")
plt.title("Fonction de densite")
plt.legend()
plt.savefig('Graph\integral.png')
plt.show()

# Calcul de la VaR

alpha = 0.1

Index = floor(alpha * Nmc)
VaR = ProfitandLoss[Index]

stats.probplot(ProfitandLoss, dist="norm", plot=pylab)
plt.savefig("Graph\quantile.png")
pylab.show()
print("Value at Risk =", VaR)
