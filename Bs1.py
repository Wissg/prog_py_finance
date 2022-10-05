import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import Bs


def F(Marche,t,S, K, T, r, sigma):
    return Bs.BS_CALL(t,So, K, T, r, sigma)-Marche

def Find_volatility_implicite(t,So, K, T, r,M,epsilon):
    sigma =[]
    vol = []

    for i in range(0,len(K)):
        if M[i] < So and M[i] > max(So-K[i]*np.exp(-r*T),0):
            sigma.append(np.sqrt(2*np.abs(np.log(So/K[i]) + r*T)/T))
            while np.abs(F(M[i],t,So,K[i],T,r,sigma[len(sigma)-1])) > epsilon :
                sigma.append(sigma[len(sigma)-1] - F(M[i],t,So,K[i],T,r,sigma[len(sigma)-1]) / Bs.Vega(t,So, K[i], T, r, sigma[len(sigma)-1]))
            vol.append(sigma[len(sigma)-1])
            sigma=[]
        else:
            vol.append(0)
            sigma = []
    return vol


def condition_arbitrage(Vol,K,M):
    ii = np.where(np.array(Vol) == 0)[0]
    for i in range(0,len(ii)):
        Vol.pop(i)
        K.pop(i)
        M.pop(i)

    plt.plot(K, Vol, label='Vol implicite sans zero')
    plt.xlabel('K')
    plt.ylabel('Sigma')
    plt.legend()
    plt.show()

K=[5125,5225,5325,5425,5525,5625,5725,5825]
M=[475,405,340,280.5,226,179.5,139,105]
So=5430.3
T=4/12
r=0.05
t=0
epsilon =0.0001
Vol = Find_volatility_implicite(t,So, K, T, r,M,epsilon)
condition_arbitrage(Vol,K,M)

