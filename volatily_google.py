import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs


data = pd.read_excel("data\GoogleOrig.xlsx")
print(data)
r= 0.006
t=0
epsilon=0.0001
S = 591.66

M = data.iloc[:,2]
K = data["Strike"]
T = data["Time"]
data["sigmaC"]=0



for i in range(0, len(M)):
    data.loc[i,"sigmaC"] = Bs.Find_volatility_implicite_fixe_Call(t, S, K[i], T[i], r, M[i], epsilon)

data = data[data.sigmaC != 0]
plt.plot(data.loc[:49,"Strike"], data.loc[:49,"sigmaC"], label='Vol implicite Call')
plt.xlabel('K')
plt.ylabel('Sigma')
plt.legend()
plt.show()
print(data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:,"Time"],data.loc[:,"Strike"],data.loc[:,"sigmaC"])
plt.show()

