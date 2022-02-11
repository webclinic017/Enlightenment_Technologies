from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pandas_datareader import data

from fracdiff.sklearn import FracdiffStat
from fracdiff.sklearn import Fracdiff
from fracdiff import fdiff

from matplotlib.pyplot import figure

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

frac_diff_value = .95
frac_int_value = -frac_diff_value

x = np.arange(0,10*np.pi,0.01)   # start,stop,step
y = np.sin(x)

y_diff = fdiff(y, n = frac_diff_value)
y_int = fdiff(y_diff, n = frac_int_value)

figure(figsize=(10, 6))
plt.plot(x,y, label = 'f(x)')
plt.plot(x,y_diff, label = 'FracDiff of f(x) by {}'.format(frac_diff_value))
plt.plot(x,y_int, label = 'Int of FracDiff by {}'.format(frac_int_value))
plt.legend()
plt.show()



























