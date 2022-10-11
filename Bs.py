import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

N = norm.cdf


def BS_CALL(t, S, K, T, r, sigma):
    if t == T:
        return max(S - K, 0)
    else:
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt((T - t)))
        d2 = d1 - sigma * np.sqrt((T - t))
        return S * N(d1) - K * np.exp(-r * (T - t)) * N(d2)


def BS_PUT(t, S, K, T, r, sigma):
    if t == T:
        return max(K - S, 0)
    else:
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt((T - t)))
        d2 = d1 - sigma * np.sqrt((T - t))
        return BS_CALL(t, S, K, T, r, sigma) - S + K * np.exp(-r * T)


def Vega(t, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt((T - t)))
    return S * np.sqrt(T - t) / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 / 2)


def Delta(t, S, K, T, r, sigma):
    if t == T:
        return 1
    else:
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt((T - t)))
    return N(d1)


def test():
    S = np.zeros(101)
    S[0] = 1
    calls = []
    vegas = []
    calls.append(BS_CALL(0, S[0], 10, 1, 0.1, 0.5))
    vegas.append(Vega(0, S[0], 10, 1, 0.1, 0.5))
    for i in range(1, 101, 1):
        S[i] = 0.2 * i
        calls.append(BS_CALL(0, S[i], 10, 1, 0.1, 0.5))
        vegas.append(Vega(0, S[i], 10, 1, 0.1, 0.5))

    plt.plot(S, calls, label='Call Value')
    plt.plot(S, vegas, label='Vega')
    plt.xlabel('$S_0$')
    plt.ylabel(' Value')
    plt.legend()
    plt.show()


def F_Call(Marche, t, S, K, T, r, sigma):
    return BS_CALL(t, S, K, T, r, sigma) - Marche


def F_Put(Marche, t, S, K, T, r, sigma):
    return BS_PUT(t, S, K, T, r, sigma) - Marche


def Find_volatility_implicite_fixe_Call(t, S, K, T, r, M, epsilon):
    if M < S and M > max(S - K * np.exp(-r * T), 0):
        sigma = np.sqrt(2 * np.abs((np.log(S / K) + r * T) / T))
        while np.abs(F_Call(M, t, S, K, T, r, sigma)) > epsilon:
            sigma = sigma - (F_Call(M, t, S, K, T, r, sigma) / Vega(t, S, K, T, r, sigma))
        vol = sigma
    else:
        "condition d'absence d'arbitrage"
        vol = 0
    return vol


def Find_volatility_implicite_fixe_Put(t, S, K, T, r, M, epsilon):
    sigma = []
    vol = []

    if M < K * np.exp(-r * T) and M > max(K * np.exp(-r * T) - S, 0):
        sigma.append(np.sqrt(2 * np.abs(np.log(S / K) + r * T) / T))
        while np.abs(F_Put(M, t, S, K, T, r, sigma[len(sigma) - 1])) > epsilon:
            sigma.append(sigma[len(sigma) - 1] - (
                    F_Put(M, t, S, K, T, r, sigma[len(sigma) - 1]) / Vega(t, S, K, T, r, sigma[len(sigma) - 1])))
        vol.append(sigma[len(sigma) - 1])
        sigma = []
    else:
        "condition d'absence d'arbitrage"
        vol.append(0)
        sigma = []
    return vol
