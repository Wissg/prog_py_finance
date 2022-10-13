import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from mpl_toolkits import mplot3d
import Bs

data = pd.read_csv("data\spx_quotedata.csv", delimiter=",")
print(data)
