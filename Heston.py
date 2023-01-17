import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def Heston(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0):
    S = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S[0] = S0
    V[0] = V0
    deltat = T / N
    t = np.linspace(0, T, N + 1)
    payoff = np.zeros(Nmc)
    log_return = np.zeros(Nmc)
    evol_Actif = np.zeros((Nmc, N + 1))
    evol_Vol = np.zeros((Nmc, N + 1))

    for p in range(0, Nmc):
        evol_Actif[p, 0] = S[0]
        evol_Vol[p, 0] = V[0]
        for i in range(0, N):
            N1 = np.random.normal()
            N2 = np.random.normal()
            S[i + 1] = S[i] * np.exp((r - V[i] / 2) * deltat + np.sqrt(V[i]) * (
                    rho * np.sqrt(deltat) * N1 + np.sqrt(1 - rho ** 2) * np.sqrt(deltat) * N2))
            V[i + 1] = V[i] + k * (theta - V[i]) * deltat + etha * np.sqrt(V[i]) * np.sqrt(
                deltat) * N1 + (etha ** 2) / 4 * deltat * (N1 ** 2 - 1)
            evol_Actif[p, i + 1] = S[i + 1]
            evol_Vol[p, i + 1] = V[i + 1]
        payoff[p] = np.maximum(S[N - 1] - K, 0)
        log_return[p] = np.log(S[N - 1] / S0)
    V_heston = np.sum(payoff) / Nmc
    plt.show()
    return log_return, evol_Vol, evol_Actif


def Heston_Estimateur1(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0):
    S = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S[0] = S0
    V[0] = V0
    deltat = T / N
    t = np.linspace(0, T, N + 1)
    payoff = np.zeros(Nmc)
    log_return = np.zeros(Nmc)
    evol_Actif = np.zeros((Nmc, N + 1))
    evol_Vol = np.zeros((Nmc, N + 1))

    for p in range(0, Nmc):
        evol_Actif[p, 0] = S[0]
        evol_Vol[p, 0] = V[0]
        for i in range(0, N):
            N1 = np.random.normal()
            N2 = np.random.normal()
            S[i + 1] = S[i] * np.exp((r - V[i] / 2) * deltat + np.sqrt(V[i]) * (
                    rho * np.sqrt(deltat) * N1 + np.sqrt(1 - rho ** 2) * np.sqrt(deltat) * N2))
            V[i + 1] = V[i] + k * (theta - V[i]) * deltat + etha * np.sqrt(V[i]) * np.sqrt(
                deltat) * N1 + (etha ** 2) / 4 * deltat * (N1 ** 2 - 1)
            evol_Actif[p, i + 1] = S[i + 1]
            evol_Vol[p, i + 1] = V[i + 1]
        payoff[p] = np.maximum(S[N - 1] - K, 0)
        log_return[p] = np.log(S[N - 1] / S0)
    V_heston = np.sum(payoff) * np.exp(-r * T) / Nmc
    plt.show()
    return V_heston


def Heston_Estimateur2(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0):
    S = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S_Symetrique = np.zeros(N + 1)
    V_Symetrique = np.zeros(N + 1)
    S[0] = S0
    V[0] = V0
    S_Symetrique[0] = S0
    V_Symetrique[0] = V0
    deltat = T / N
    t = np.linspace(0, T, N + 1)
    payoff = np.zeros(Nmc)
    log_return = np.zeros(Nmc)
    evol_Actif = np.zeros((Nmc, N + 1))
    evol_Vol = np.zeros((Nmc, N + 1))

    for p in range(0, Nmc):
        evol_Actif[p, 0] = S[0]
        evol_Vol[p, 0] = V[0]
        for i in range(0, N):
            N1 = np.random.normal()
            N2 = np.random.normal()
            S[i + 1] = S[i] * np.exp((r - V[i] / 2) * deltat + np.sqrt(V[i]) * (
                    rho * np.sqrt(deltat) * N1 + np.sqrt(1 - rho ** 2) * np.sqrt(deltat) * N2))
            V[i + 1] = V[i] + k * (theta - V[i]) * deltat + etha * np.sqrt(V[i]) * np.sqrt(
                deltat) * N1 + (etha ** 2) / 4 * deltat * (N1 ** 2 - 1)

            S_Symetrique[i + 1] = S_Symetrique[i] * np.exp(
                (r - V_Symetrique[i] / 2) * deltat + np.sqrt(V_Symetrique[i]) * (
                        rho * np.sqrt(deltat) * -N1 + np.sqrt(1 - rho ** 2) * np.sqrt(deltat) * -N2))
            V_Symetrique[i + 1] = V_Symetrique[i] + k * (theta - V_Symetrique[i]) * deltat + etha * np.sqrt(
                V_Symetrique[i]) * np.sqrt(
                deltat) * -N1 + (etha ** 2) / 4 * deltat * (N1 ** 2 - 1)

            evol_Actif[p, i + 1] = S[i + 1]
            evol_Vol[p, i + 1] = V[i + 1]
        payoff[p] = np.maximum(S[N - 1] - K, 0) + np.maximum(S_Symetrique[N - 1] - K, 0)
        log_return[p] = np.log(S[N - 1] / S0)

    V_heston = np.sum(payoff) * np.exp(-r * T) / (2 * Nmc)
    return V_heston


def Heston_Estimateur2_Matrice(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0, N11, N22):
    S = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S_Symetrique = np.zeros(N + 1)
    V_Symetrique = np.zeros(N + 1)
    S[0] = S0
    V[0] = V0
    S_Symetrique[0] = S0
    V_Symetrique[0] = V0
    deltat = T / N
    t = np.linspace(0, T, N + 1)
    payoff = np.zeros(Nmc)
    log_return = np.zeros(Nmc)
    evol_Actif = np.zeros((Nmc, N + 1))
    evol_Vol = np.zeros((Nmc, N + 1))

    for p in range(0, Nmc):
        evol_Actif[p, 0] = S[0]
        evol_Vol[p, 0] = V[0]
        for i in range(0, N):
            N1 = N11[p, i]
            N2 = N22[p, i]
            S[i + 1] = S[i] * np.exp((r - V[i] / 2) * deltat + np.sqrt(V[i]) * (
                    rho * np.sqrt(deltat) * N1 + np.sqrt(1 - rho ** 2) * np.sqrt(deltat) * N2))
            V[i + 1] = V[i] + k * (theta - V[i]) * deltat + etha * np.sqrt(V[i]) * np.sqrt(
                deltat) * N1 + (etha ** 2) / 4 * deltat * (N1 ** 2 - 1)

            S_Symetrique[i + 1] = S_Symetrique[i] * np.exp(
                (r - V_Symetrique[i] / 2) * deltat + np.sqrt(V_Symetrique[i]) * (
                        rho * np.sqrt(deltat) * -N1 + np.sqrt(1 - rho ** 2) * np.sqrt(deltat) * -N2))
            V_Symetrique[i + 1] = V_Symetrique[i] + k * (theta - V_Symetrique[i]) * deltat + etha * np.sqrt(
                V_Symetrique[i]) * np.sqrt(
                deltat) * -N1 + (etha ** 2) / 4 * deltat * (N1 ** 2 - 1)

            evol_Actif[p, i + 1] = S[i + 1]
            evol_Vol[p, i + 1] = V[i + 1]
        payoff[p] = np.maximum(S[N - 1] - K, 0) + np.maximum(S_Symetrique[N - 1] - K, 0)
        log_return[p] = np.log(S[N - 1] / S0)

    V_heston = np.sum(payoff) * np.exp(-r * T) / (2 * Nmc)
    return V_heston


def Etha_Hetson(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0, h2, N11, N22):
    return (Heston_Estimateur2_Matrice(Nmc, N, K, S0, T, r, k, rho, theta, etha + h2, V0, N11,
                                       N22) - Heston_Estimateur2_Matrice(Nmc, N, K, S0, T,
                                                                         r, k, rho,
                                                                         theta, etha - h2,
                                                                         V0, N11, N22)) / (2 * h2)


def theta_Hetson(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0, h1, N11, N22):
    return (Heston_Estimateur2_Matrice(Nmc, N, K, S0, T, r, k, rho, theta + h1, etha, V0, N11,
                                       N22) - Heston_Estimateur2_Matrice(Nmc, N, K, S0, T,
                                                                         r, k, rho,
                                                                         theta - h1, etha,
                                                                         V0, N11, N22)) / (2 * h1)


def LevenbergMarquard(Nmc, N, S0, T, r, k, rho, theta, etha, V0, epsilon, lamb, Kp, Vp, N11, N22):
    d = [1, 1]
    count = 0
    res = np.zeros(len(Kp))
    Jacobien = np.zeros((len(Kp), 2))
    while np.linalg.norm(d, 2) > epsilon:
        count = count + 1
        for i in range(len(Kp)):
            Vheston = Heston_Estimateur2(Nmc, N, Kp[i], S0, T, r, k, rho, theta, etha, V0)
            res[i] = Vp[i] - Vheston
            Jacobien[i, 0] = -theta_Hetson(Nmc, N, Kp[i], S0, T, r, k, rho, theta, etha, V0, h2, N11, N22)
            Jacobien[i, 1] = -Etha_Hetson(Nmc, N, Kp[i], S0, T, r, k, rho, theta, etha, V0, h2, N11, N22)

        Hesienne = (np.dot(Jacobien.T, Jacobien) + lamb * np.identity(2))
        d = -np.dot(np.linalg.inv(np.dot(Jacobien.T, Jacobien) + lamb * np.identity(2)),
                    np.dot(Jacobien.T, res))
        theta = theta + d[0]
        etha = etha + d[1]
        if theta > 1 or theta < 0:
            theta = 0.2
        if etha > 1 or etha < 0:
            etha = 0.5

        print(theta)
        print(etha)
        print("count = ", count)
    print("d = ", d)
    print("Hesienne = ", Hesienne)
    return theta, etha


N = 100
Nmc = 10000
K = 1
S0 = 1
T = 0.5
r = 0.01
k = 2
rho = 0.9
theta = 0.04
etha = 0.3
V0 = 0.04
# log_return1, evol_Vol, evol_Actif = Heston(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0)
# for i in range(0, 4):
#     plt.plot(evol_Vol[i, :])
#     plt.title("Evolution de la volatilité rho = 0.9")
# plt.savefig('Graph\Evolution_volatilié_rho_-0-9.png')
# plt.show()
# for i in range(0, 4):
#     plt.plot(evol_Actif[i, :])
#     plt.title("Evolution de L'actif rho = 0.9")
# plt.savefig('Graph\Evolution_actif_rho_-0-9.png')
# plt.show()
#
#
# rho = 0
# log_return2, evol_Vol, evol_Actif = Heston(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0)
# for i in range(0, 4):
#     plt.plot(evol_Vol[i, :])
#     plt.title("Evolution de la volatilité rho = 0")
# plt.savefig('Graph\Evolution_volatilité_rho_0.png')
# plt.show()
# for i in range(0, 4):
#     plt.plot(evol_Actif[i, :])
#     plt.title("Evolution de L'actif rho = 0")
# plt.savefig('Graph\Evolution_actif_rho_0.png')
# plt.show()
#
# rho = -0.9
# log_return3, evol_Vol, evol_Actif = Heston(Nmc, N, K, S0, T, r, k, rho, theta, etha, V0)
# for i in range(0, 4):
#     plt.plot(evol_Vol[i, :])
#     plt.title("Evolution de la volatilité rho = -0.9")
# plt.savefig('Graph\Evolution_volatilité_rho_-0-9.png')
# plt.show()
# for i in range(0, 4):
#     plt.plot(evol_Actif[i, :])
#     plt.title("Evolution de L'actif rho = -0.9")
# plt.savefig('Graph\Evolution_actif_rho_-0-9.png')
# plt.show()
#
# sns.kdeplot(log_return1, label="rho = 0.9")
# sns.kdeplot(log_return2, label="rho = 0")
# sns.kdeplot(log_return3, label="rho = -0.9")
# plt.xlabel("log return")
# plt.ylabel("Valeur")
# plt.title("Fonction de densite log return")
# plt.legend()
# plt.savefig('Graph\densite_log_return.png')
# plt.show()
Nmc = 10
K = 10
# S = np.linspace(0.01, 20, 500)
# VH1 = np.zeros(len(S))
# VH2 = np.zeros(len(S))
# for i in range(len(S)):
#     VH1[i] = Heston_Estimateur1(Nmc, N, K, S[i], T, r, k, rho, theta, etha, V0)
#     VH2[i] = Heston_Estimateur2(Nmc, N, K, S[i], T, r, k, rho, theta, etha, V0)
#
# plt.plot(S, VH1, label="Estimateur 1")
# plt.plot(S, VH2, label="Estimateur 2")
# plt.xlabel("S0")
# plt.ylabel("V Heston")
# plt.title("C")
# plt.legend()
# plt.savefig('Graph\V_Heston-Estimateur.png')
# plt.show()

Nmc = 1000
N = 100
S0 = 10
h1 = 0.1
h2 = h1
T = 0.5
r = 0.1
k = 3
rho = 0.5
theta = 0.2
etha = 0.5
V0 = 0.04
# K1 = np.linspace(0.1, 20, 40)
# Etha1 = np.zeros(len(K1))
# Theta1 = np.zeros(len(K1))
# N11 = np.zeros((Nmc, N))
# N22 = np.zeros((Nmc, N))
# for p in range(0, Nmc):
#     for i in range(0, N):
#         N11[p, i] = np.random.normal()
#         N22[p, i] = np.random.normal()
#
# for i in range(len(K1)):
#     Etha1[i] = Etha_Hetson(Nmc, N, K1[i], S0, T, r, k, rho, theta, etha, V0, h1, N11, N22)
#     Theta1[i] = theta_Hetson(Nmc, N, K1[i], S0, T, r, k, rho, theta, etha, V0, h1, N11, N22)
# plt.plot(K1, Etha1)
# plt.xlabel("K")
# plt.ylabel("Eta Heston")
# plt.title("Eta Heston")
# plt.savefig('Graph\Etha_Heston.png')
# plt.show()
# plt.plot(K1, Theta1)
# plt.xlabel("K")
# plt.ylabel("Theta Heston")
# plt.title("Theta Estimateur")
# plt.savefig('Graph\Theta_Heston.png')
# plt.show()

Nmc = 100
theta = 0.2
etha = 0.5
epsilon = 0.0001
lamb = 0.01
S0 = 10
r = 0.01
V0 = 0.04
rho = 0.5
k = 3

Vm = [2.0944, 1.7488, 1.4266, 1.1456, 0.8919, 0.7068, 0.5461, 0.4187, 0.3166, 0.2425, 0.1860, 0.1370, 0.0967, 0.0715,
      0.0547, 0.0381, 0.0306, 0.0239, 0.0163, 0.0139, 0.086]
Kp = np.arange(8, 16, 0.4)

N11 = np.zeros((Nmc, N))
N22 = np.zeros((Nmc, N))
for p in range(0, Nmc):
    for i in range(0, N):
        N11[p, i] = np.random.normal()
        N22[p, i] = np.random.normal()

theta, etha = LevenbergMarquard(Nmc, N, S0, T, r, k, rho, theta, etha, V0, epsilon, lamb, Kp, Vm, N11, N22)
print("theta = ", theta, " eta =", etha)
