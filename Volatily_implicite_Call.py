import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs

data = pd.read_csv("data\sp-index.txt", delimiter="\s+")

So = 1260.36
q = 0.0217
data["S"] = np.exp(-q*data.iloc[:, 0])*So
data["MarcheC"] = (data.iloc[:, 3] + data.iloc[:, 2]) / 2

epsilon = 0.0001
t = 0
S = data["S"]
M = data["MarcheC"]
K = data["K"]
r = data["r"] / 100
T = data["T"]
data["sigmaC"] = 0
data["sigmaP"] = 0


for i in range(0, len(S)):
    data.loc[i, "sigmaC"] = Bs.Find_volatility_implicite_fixe_Call(t, S[i], K[i], T[i], r[i], M[i], epsilon)

data = data[data.sigmaC != 0]
plt.plot(data.loc[:57, "K"], data.loc[:57, "sigmaC"], label='Vol implicite Call')

plt.xlabel('K')
plt.ylabel('Sigma')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, "T"], data.loc[:, "K"], data.loc[:, "sigmaC"])
ax.set_xlabel('T')
ax.set_ylabel('K')
ax.set_zlabel('sigma')
plt.show()

