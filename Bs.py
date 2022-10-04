import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

N = norm.cdf

def BS_CALL(t,S, K, T, r, sigma):
    if t==T:
        return max(S-K,0)
    else:
        d1 = (np.log(S/K) + (r + sigma**2/2)*(T-t)) / (sigma*np.sqrt((T-t)))
        d2 = d1 - sigma * np.sqrt((T-t))
        return S * N(d1) - K * np.exp(-r*(T-t))* N(d2)

def BS_PUT(t,S, K, T, r, sigma):
    if t==T:
        return max(S-K,0)
    else:
        d1 = (np.log(S/K) + (r + sigma**2/2)*(T-t)) / (sigma*np.sqrt((T-t)))
        d2 = d1 - sigma* np.sqrt((T-t))
        return K*np.exp(-r*(T-t))*N(-d2) - S*N(-d1)

def Vega(t,S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*(T-t)) / (sigma * np.sqrt((T-t)))
    return S * np.sqrt(T-t) / np.sqrt(2 * np.pi)* np.exp(-d1**2/2)

S=np.zeros(101)
S[0]=1
calls=[]
vegas=[]
calls.append(BS_CALL(0,S[0],10,1,0.1,0.5))
vegas.append(Vega(0,S[0],10,1,0.1,0.5))
for i in range(1,101,1):
    S[i]= 0.2*i
    calls.append(BS_CALL(0,S[i],10,1,0.1,0.5))
    vegas.append(Vega(0, S[i], 10, 1, 0.1, 0.5))


plt.plot(S,calls, label='Call Value')
plt.plot(S,vegas, label='Vega')
plt.xlabel('$S_0$')
plt.ylabel(' Value')
plt.legend()
plt.show()






