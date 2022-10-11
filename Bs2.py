import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs

data = pd.read_csv("h:\Downloads\sp-index.txt", delimiter="\s+")

So = 1226
q = 0.0217
data["S"] = So * np.exp(-q * data.iloc[:, 0])
data["MarcheC"] = (data.iloc[:, 3] + data.iloc[:, 2]) / 2
data["MarcheP"] = (data.iloc[:, 5] + data.iloc[:, 4]) / 2

epsilon = 0.0001
t = 0
S = data["S"]
M = data["MarcheC"]
MP = data["MarcheP"]
K = data["K"]
r = data["r"] / 100
T = data["T"]
data["sigmaC"] = 0
data["sigmaP"] = 0


for i in range(0, 2):
    data.loc[i, "sigmaC"] = Bs.Find_volatility_implicite_fixe_Call(t, S[i], K[i], T[i], r[i], M[i], epsilon)
    data.loc[i, "sigmaP"] = Bs.Find_volatility_implicite_fixe_Put(t, S[i], K[i], T[i], r[i], MP[i], epsilon)

plt.plot(data.loc[:57, "K"], data.loc[:57, "sigmaC"], label='Vol implicite Call')

plt.xlabel('K')
plt.ylabel('Sigma')
plt.legend()
plt.show()

plt.plot(data.loc[:57, "K"], data.loc[:57, "sigmaP"], label='Vol implicite Put')

plt.xlabel('K')
plt.ylabel('Sigma')
plt.legend()
plt.show()

data = data[data.sigmaC != 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, "T"], data.loc[:, "K"], data.loc[:, "sigmaC"])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, "T"], data.loc[:, "K"], data.loc[:, "sigmaP"])
plt.show()

print(data)
