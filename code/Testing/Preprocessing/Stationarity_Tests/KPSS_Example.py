from statsmodels.tsa.stattools import kpss

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# From https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/

# The KPSS test, short for, Kwiatkowski-Phillips-Schmidt-Shin (KPSS), is a type of Unit root test 
# that tests for the stationarity of a given series around a deterministic trend. In other words, 
# the test is somewhat similar in spirit with the ADF test. A common misconception, however, is that 
# it can be used interchangeably with the ADF test. This can lead to misinterpretations about the 
# stationarity, which can easily go undetected causing more problems down the line. 


# A key difference from ADF test is the null hypothesis of the KPSS test is that the 
# series is stationary. So practically, the interpretaion of p-value is just the opposite 
# to each other. That is, if p-value is < signif level (say 0.05), then the series is non-stationary. 
# Whereas in ADF test, it would mean the tested series is stationary.



#%%

# KPSS Test on Known Non-Stationary Time-Series w/ Seasonality

# Load the dataset
df = sm.datasets.sunspots.load_pandas().data
df.shape
print("Dataset has {} records and {} columns".format(df.shape[0], df.shape[1]))
df['YEAR'] = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
df.index = df['YEAR']
del df['YEAR']
df.head()
plt.figure(figsize=(16,5))

# Plot the data
plt.plot(df.index, df['SUNACTIVITY'], label = "SUNACTIVITY")
plt.legend(loc='best')
plt.title("Sunspot Data from year 1700 to 2008")
plt.show()

series = df.loc[:, 'SUNACTIVITY'].values
series = series[:, np.newaxis]

def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    
# Call the function and run the test
kpss_test(series)


#%%

# KPSS Test on Known Stationary Time-Series

# KPSS test on random numbers
series = np.random.randn(100)
a_list = list(range(0, 100))

kpss_test(series)
    
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(series);
plt.title('Random')













