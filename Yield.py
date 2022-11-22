import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats  # needed for Phi
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import pylab
import Bs


def B(t, T, gamma):
    return (1 - np.exp(-gamma * (T - t))) / gamma


def A(t, T, gamma, etha, sigma):
    return ((B(t, T, gamma) - T + t) * (etha * gamma - sigma ** 2 / 2) / (gamma ** 2)) - (
            sigma ** 2 * B(t, T, gamma) ** 2) / (4 * gamma)


def B_derivative(gamma, t, T):
    return ((T - t) * np.exp(-gamma * (T - t)) - B(t, T, gamma)) / gamma


def A_derivative(gamma, sigma, etha, t, T):
    return (etha * (B_derivative(gamma, t, T) * gamma - B(t, T, gamma)) + (T - t) * etha - 0.5 * sigma ** 2 * (
            B_derivative(gamma, t, T) - 2 * B(t, T, gamma) / gamma) - (
                    (T - t) / gamma) * sigma ** 2 - 0.25 * sigma ** 2 * B(t, T, gamma) * (
                    2 * gamma * B_derivative(gamma, t, T) - B(t, T, gamma))) / gamma ** 2


def Vavisek(t, T, r0, gamma, etha, sigma):
    return -np.log(np.exp(A(t, T, gamma, etha, sigma) - r0 * B(t, T, gamma))) / (T - t)


def Yield(t, T, r0, gamma, etha, sigma):
    return (-A(t, T, gamma, etha, sigma) + r0 * B(t, T, gamma)) / (T - t)


def Calibration_yield(Ym, T, epsilon, t, r, lamb, etha, gamma, sigma):
    Res = [0] * len(Ym)
    Yth = [0] * len(Ym)
    Jacobien = np.zeros(shape=(10, 3))
    d = [1, 1, 1]
    while np.linalg.norm(d, 2) > epsilon:
        for p in range(len(T)):
            tau = T[p] - t
            Yth[p] = Yield(t, T[p], r, gamma, etha, np.sqrt(sigma))
            Res[p] = Ym[p] - Yth[p]
            Jacobien[p, 0] = (B(t, T[p], gamma) - tau) / (gamma * tau)
            Jacobien[p, 1] = -1 * ((1 / (tau * gamma)) * (
                    (B(t, T[p], gamma) - tau) / (2 * gamma) + ((B(t, T[p], gamma) ** 2) / 4)))
            Jacobien[p, 2] = 1 / tau * (
                    A_derivative(gamma, np.sqrt(sigma), etha, t, T[p]) - r * B_derivative(gamma, t, T[p]))

        d = -np.dot(np.linalg.inv(np.dot(Jacobien.T, Jacobien) + lamb * np.identity(3)),
                    np.dot(Jacobien.T, Res))
        etha = etha + d[0]
        sigma = sigma + d[1]
        gamma = gamma + d[2]
    print('etha = ', etha, ' sigma =', sigma, ' gamma = ', gamma)
    return Yth


def Calibrition_historical(N, T, epsilon, lamb, etha, gamma, sigma):
    r = np.zeros(N)
    delta_t = T / N

    Res = [0] * N
    Jacobien = np.zeros(shape=(N, 2))
    for i in range(N - 1):
        r[i + 1] = r[i] * np.exp(-gamma * delta_t) + etha / gamma * (1 - np.exp(-gamma * delta_t)) + sigma * (
            np.sqrt((1 - np.exp(-2 * gamma * delta_t)) / (2 * gamma))) * np.random.normal()

    a = np.exp(-gamma * delta_t)
    b = etha / gamma * (1 - np.exp(-gamma * delta_t))
    d = [a, b]
    while np.linalg.norm(d, 2) > epsilon:
        for p in range(N - 1):
            Res[p] = r[i + 1] - a * r[i] - b
            Jacobien[p, 0] = -r[p]
            Jacobien[p, 1] = -1

        d = -np.dot(np.linalg.inv(np.dot(Jacobien.T, Jacobien) + lamb * np.identity(2)),
                    np.dot(Jacobien.T, Res))
        a = a + d[0]
        b = b + d[1]

    D = [r_i_1 - a * r_i - b for r_i_1, r_i in zip(r[1:], r[:N])]
    gamma = -np.log(a) / delta_t
    etha = gamma * (b / (1 - a))
    sigma = np.std(D) * np.sqrt((-2 * np.log(a) / (delta_t * (1 - a ** 2))))
    print("Variance is equal to ", np.std(D) ** 2)

    plt.scatter(r[:N - 1], r[1:])
    y = a * r + b
    plt.plot(r, y, label="linear regression")
    plt.xlabel("r i")
    plt.ylabel("r i+1")
    plt.title("Vasicek interest rate")
    plt.legend()
    plt.savefig('Graph\Vasicek_interest_rate.png')
    plt.show()

    print('etha = ', etha, ' sigma =', sigma, ' gamma = ', gamma)
    print('a = ', a, 'b = ', b)
    return etha, gamma, sigma

# Exo 1
t = 0
r0 = 0.027
N = 10000
T = np.linspace(0.001, 30, N + 1)
gamma = 0.25
etha = 0.25 * 0.03
sigma = 0.02
lamb = 0.01
epsilon = 10 ** (-9)
p = np.zeros(N + 1)
# for i in range(0, N + 1):
#     p[i] = Yield(t, T[i], r0, gamma, etha, sigma)
# plt.plot(T, p)
# plt.axhline(y=r0, color='r', linestyle='--',label="r0 = 0.027")
# plt.xlabel("Maturity")
# plt.ylabel("Yields")
# plt.title("Yields Slightly humped")
# plt.legend()
# plt.savefig('Graph\Yields.png')
# plt.show()
#
# r0 = 0.01
# for i in range(0, N + 1):
#     p[i] = Yield(t, T[i], r0, gamma, etha, sigma)
# plt.plot(T, p)
# plt.axhline(y=r0, color='r', linestyle='--',label="r0 = 0.01")
# plt.xlabel("Maturity")
# plt.ylabel("Yields")
# plt.title("Yields upward sloping")
# plt.legend()
# plt.savefig('Graph\Yields1.png')
# plt.show()


N = 1000000
p = np.zeros(N + 1)
# T = np.linspace(100, 0, N + 1,endpoint=False)
# r0 = 0.05
# for i in range(0, N + 1):
#     p[i] = Yield(t, T[i], r0, gamma, etha, sigma)
# plt.plot(p, T)
# plt.axhline(y=r0, color='r', linestyle='--',label="r0 = 0.05")
# plt.xlabel("Yields")
# plt.ylabel("Maturity")
# plt.title("Yields curved")
# plt.legend()
# plt.savefig('Graph\Lim_Yields_zeros.png')
# plt.show()
# print("r0 = ",r0," Y(0,T) T->0 = ",p[N])

T = np.linspace(1, 9999, N + 1)
for i in range(N + 1):
    p[i] = Yield(t, T[i], r0, gamma, etha, sigma)
plt.plot(T, p)
plt.axhline(y=etha/gamma - 0.5 * (sigma/gamma)**2, color='r', linestyle='--',label="etha/gamma - 0.5 * (sigma/gamma)^2")
plt.xlabel("Maturity")
plt.ylabel("Yields")
plt.title("Lim Y(0,T) T->+infinite")
plt.legend()
plt.savefig('Graph\Lim_Yields_inf.png')
plt.show()

print("etha/gamma - 0.5 * (sigma/gamma)^2 = ",etha/gamma - 0.5 * (sigma/gamma)**2," Y(0,T) T->+infinite = ",p[N])
# Exo 2

N = 10000
gamma = 0.25
etha = 0.25 * 0.03
sigma = 0.02
lamb = 0.01
epsilon = 10 ** (-9)
r0 = 0.027
t =0

Ym = [0.035, 0.041, 0.0439, 0.046, 0.0484, 0.0494, 0.0507, 0.0517, 0.052, 0.0523]
T = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
Yth = Calibration_yield(Ym, T, epsilon, t, r0, lamb, etha, gamma, sigma)

plt.scatter(T, Ym, label='Market Yields')
plt.plot(T, Yth, label='Vasicek Yields')
plt.xlabel("Maturity")
plt.ylabel("Yields")
plt.title("Yields Curved")
plt.legend()
plt.savefig('Graph\Curved_Yields.png')
plt.show()

# Exo 3
N = 10000
Y_m = [0.056, 0.064, 0.074, 0.081, 0.082, 0.09, 0.087, 0.092, 0.0895, 0.091]
gamma = 0.25
etha = 0.25 * 0.03
sigma = 0.02
lamb = 0.01
epsilon = 10 ** (-9)
r0 = 0.04
t = 1
T = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
Yth = Calibration_yield(Ym, T, epsilon, t, r0, lamb, etha, gamma, sigma)
plt.scatter(T, Ym, label='Market Yields')
plt.plot(T, Yth, label='Vasicek Yields')
plt.xlabel("Maturity")
plt.ylabel("Yields")
plt.title("Yields Curved")
plt.legend()
plt.savefig('Graph\Curved_Yields1.png')
plt.show()

# Exo 4
# T = 5
# gamma = 4
# etha = 0.6
# sigma = 0.08
# lamb = 0.01
# N = 50
# Calibrition_historical(N, T, epsilon, lamb, etha, gamma, sigma)
