from fracdiff.sklearn import FracdiffStat, Fracdiff
from fracdiff.fdiff import fdiff_coef 
from fracdiff import fdiff


import statsmodels.tsa.stattools as stattools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas_datareader

# Adopted from https://github.com/simaki/fracdiff/blob/main/sample/examples/examples.ipynb


#%%

plt.figure(figsize=(24, 6))

plt.subplot(1, 2, 1)
plt.title("Coefficients of fractional differentiation for d=0.0-1.0")
for d in np.linspace(0.0, 1.0, 5):
    plt.plot(fdiff_coef(d, 6), label=f"d={d:.2f}")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Coefficients of fractional differentiation for d=1.0-2.0")
for d in np.linspace(1.0, 2.0, 5):
    plt.plot(fdiff_coef(d, 6), label=f"d={d:.2f}")
plt.legend()

plt.show()


#%%

# S&P 500

def fetch_yahoo(ticker, begin="1998-01-01", end="2020-09-30"):
    """
    Returns: pandas.Series
    """
    return pandas_datareader.data.DataReader(ticker, "yahoo", begin, end)["Adj Close"]


def fetch_fred(ticker, begin="1998-01-01", end="2020-09-30"):
    """
    Returns: pandas.Series
    """
    return pandas_datareader.data.DataReader(ticker, "fred", begin, end).iloc[:, 0]

spx = fetch_yahoo("^GSPC")

print(spx.head())

print(spx.shape)

# Plotting FracDiff
plt.figure(figsize=(24, 24))
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

for i, d in enumerate(np.linspace(0.1, 0.9, 9)):
    diff = fdiff(spx, d, mode="valid")
    diff = pd.Series(diff, index=spx.index[-diff.size :])
    plt.subplot(9, 1, i + 1)
    plt.title(f"S&P 500, {d:.1f}th differentiated")
    plt.plot(diff, linewidth=0.4)

plt.show()

#%%

# Stationarity of Fracdiff
def adfstat(d):
    diff = fdiff(spx, d, mode="valid")
    stat, *_ = stattools.adfuller(diff)
    return stat


def correlation(d):
    diff = fdiff(spx, d, mode="valid")
    corr = np.corrcoef(spx[-diff.size :], diff)[0, 1]
    return corr


ds = np.linspace(0.0, 1.0, 10)
stats = np.vectorize(adfstat)(ds)
corrs = np.vectorize(correlation)(ds)

# 5% critical value of stationarity
_, _, _, _, crit, _ = stattools.adfuller(spx)

# plot
fig, ax_stat = plt.subplots(figsize=(24, 8))
ax_corr = ax_stat.twinx()

ax_stat.plot(ds, stats, color="blue", label="ADF statistics (left)")
ax_corr.plot(ds, corrs, color="orange", label="correlation (right)")
ax_stat.axhline(y=crit["5%"], linestyle="--", color="k", label="5% critical value")

plt.title("Stationarity and memory of fractionally differentiated S&P 500")
fig.legend()
plt.show()

#%% 

# Differentiation While Preserving Memory

X = spx.values.reshape(-1, 1)

fs = FracdiffStat(mode="valid")

Xdiff = fs.fit_transform(X)
_, pvalue, _, _, _, _ = stattools.adfuller(Xdiff.reshape(-1))
corr = np.corrcoef(X[-Xdiff.size :, 0], Xdiff.reshape(-1))[0][1]

print("* Order: {:.2f}".format(fs.d_[0]))
print("* ADF p-value: {:.2f} %".format(100 * pvalue))
print("* Correlation with the original time-series: {:.2f}".format(corr))

spx_diff = pd.Series(Xdiff.reshape(-1), index=spx.index[-Xdiff.size :])

fig, ax_s = plt.subplots(figsize=(24, 6))
plt.title("S&P 500 and its differentiation preserving memory")
ax_d = ax_s.twinx()

plot_s = ax_s.plot(spx, color="blue", linewidth=0.4, label="S&P 500 (left)")
plot_d = ax_d.plot(
    spx_diff,
    color="orange",
    linewidth=0.4,
    label=f"S&P 500, {fs.d_[0]:.2f} th diff (right)",
)
plots = plot_s + plot_d

ax_s.legend(plots, [p.get_label() for p in plots], loc=0)
plt.show()

#%%

# Other Financial Data

nt_yahoo = [
    ("S&P 500", "^GSPC"),
    ("Nikkei 225", "^N225"),
    ("US 10y", "^TNX"),
    ("Apple", "AAPL"),
]
nt_fred = [
    ("USD/JPY", "DEXJPUS"),
    ("Crude Oil", "DCOILWTICO"),
]

dfy = pd.DataFrame({name: fetch_yahoo(ticker) for name, ticker in nt_yahoo})
dff = pd.DataFrame({name: fetch_fred(ticker) for name, ticker in nt_fred})

prices = pd.concat([dfy, dff], axis=1).fillna(method="ffill").loc["1998-01-05":]

print(prices)

def stats(X):
    return [stattools.adfuller(X[:, i])[0] for i in range(X.shape[1])]


ds = np.linspace(0.0, 1.0, 11)

df_stats = pd.DataFrame(
    [stats(Fracdiff(d, mode="valid").fit_transform(prices.values)) for d in ds],
    index=ds,
    columns=prices.columns,
)

print(df_stats) 

_, _, _, _, crit, _ = stattools.adfuller(prices["S&P 500"].values)

df_stats.plot(figsize=(24, 8), ylim=(-30, 5))
plt.axhline(y=crit["5%"], linestyle="--", color="gray")
plt.title("ADF statistics of fractionally differentiated prices")
plt.show()























































