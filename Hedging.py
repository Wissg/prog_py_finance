import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs

r = 0.05
K = 1.5
T = 5
sigma = 0.5
N = 100
dt = T / N
t = np.linspace(0, T, N)
S = [1] * N
A = [Bs.Delta(0, S[0], K, T, r, sigma)] * N
B = [1] * N
V = [Bs.BS_CALL(0, S[0], K, T, r, sigma)] * N
P = [A[0] * S[0] + B[0]] * N
P_actu = [V[0]] * N

for i in range(0, N - 1):
    S[i + 1] = S[i] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    A[i + 1] = Bs.Delta(t[i + 1], S[i + 1], K, T, r, sigma)
    B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * dt)
    P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
    V[i + 1] = Bs.BS_CALL(t[i + 1], S[i + 1], K, T, r, sigma)
    P_actu[i + 1] = P[i + 1] - (P[0] - V[0]) * np.exp(r * t[i + 1])

plt.plot(t, V, t, P_actu)
plt.show()
PL = V[N - 1] - P[N - 1]

plt.plot(t, A, label='A')
plt.plot(t, B, label='B')
plt.xlabel('t')
plt.ylabel('A & B')
plt.legend()
plt.show()
