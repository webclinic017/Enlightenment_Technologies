from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pandas.plotting import register_matplotlib_converters
from pandas_datareader import data

from fracdiff.sklearn import FracdiffStat
from fracdiff.sklearn import Fracdiff
from fracdiff import fdiff

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas_datareader
import seaborn


#%%

def fetch_yahoo(ticker):
    """
    Returns: pd.Series
    """

    stock_data = data.DataReader(ticker, data_source = 'yahoo', start='2015-1-1', end='2015-12-31')['Adj Close']

    return stock_data

a = np.array([1, 2, 4, 7, 0])
fdiff(a, n=0.5)
np.array_equal(fdiff(a, n=1), np.diff(a, n=1))
a = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
fdiff(a, n=0.5, axis=0)
fdiff(a, n=0.5, axis=-1)


spx = fetch_yahoo("AAPL") # S&P 500
X = spx.values.reshape(-1, 1)
f = Fracdiff(0.5, mode="valid", window=100)
X = f.fit_transform(X)
diff = pd.DataFrame(X, index=spx.index[-X.size :], columns=["SPX 0.5th fracdiff"])


fig, ax_s = plt.subplots(figsize=(24, 8))
ax_d = ax_s.twinx()

plot_s = ax_s.plot(spx, color="blue", linewidth=0.6, label="S&P 500 (left)")
plot_d = ax_d.plot(
    diff,
    color="orange",
    linewidth=0.6,
    label="S&P 500, 0.5th differentiation (right)",
)
plots = plot_s + plot_d

ax_s.legend(plots, [p.get_label() for p in plots], loc=0)
plt.title("S&P 500 and its fractional differentiation")
plt.show()


#%%
# Differentiation while preserving memory

np.random.seed(42)
X, y = np.random.randn(100, 4), np.random.randn(100)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("fracdiff", Fracdiff(0.5)),
        ("regressor", LinearRegression()),
    ]
)
pipeline.fit(X, y)


np.random.seed(42)
X = np.random.randn(100, 3).cumsum(0)
f = FracdiffStat().fit(X)
f.d_

nky = fetch_yahoo("^N225")  # Nikkei 225

fs = FracdiffStat(window=100, mode="valid")
diff = fs.fit_transform(nky.values.reshape(-1, 1))

diff = pd.DataFrame(
    diff.reshape(-1), index=nky.index[-diff.size :], columns=["Nikkei 225 fracdiff"]
)

fig, ax_s = plt.subplots(figsize=(24, 8))
ax_d = ax_s.twinx()

plot_s = ax_s.plot(spx, color="blue", linewidth=0.6, label="S&P 500 (left)")
plot_d = ax_d.plot(
    diff,
    color="orange",
    linewidth=0.6,
    label=f"Nikkei 225, {fs.d_[0]:.2f} th diff (right)",
)
plots = plot_s + plot_d

ax_s.legend(plots, [p.get_label() for p in plots], loc=0)
plt.title("Nikkei 225 and its fractional differentiation")
plt.show()



















































