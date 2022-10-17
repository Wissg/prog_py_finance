import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
from datetime import datetime
import Bs

data = pd.read_csv("data\spx_quotedata.csv", delimiter=",")

currentDate = datetime.today()
print(currentDate)
So = 3932.69

data["MarcheC"] = (data.iloc[:, 5] + data.iloc[:, 4]) / 2

epsilon = 0.0001
t = 0
M = data["MarcheC"]
K = data.iloc[:, 11]
r = 0.0255
data["sigmaC"] = 0
data["T"] = 0
j = 1
z = data.loc[164, "Expiration Date"]
for i in range(163, len(K)):
    data.loc[i, "T"] = (datetime.strptime(data.iloc[i, 0], "%d-%m-%y") - currentDate).days / 365
    if z != data.loc[i, "Expiration Date"]:
        j = j + 1
        z = data.loc[i, "Expiration Date"]
        print(i)
    data.loc[i, "T"] = j / 365.22
print(data["T"])
for i in range(163, len(M)):
    data.loc[i, "sigmaC"] = Bs.Find_volatility_implicite_fixe_Call(t, So, K[i], data.loc[i, "T"], r, M[i], epsilon)

data = data[data.sigmaC != 0]
plt.plot(data.loc[163:339, "Strike"], data.loc[163:339, "sigmaC"], label='Vol implicite Call')

plt.xlabel('K')
plt.ylabel('Sigma')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, "T"], data.loc[:, "Strike"], data.loc[:, "sigmaC"])
plt.show()
