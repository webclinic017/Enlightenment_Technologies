from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

# ADF Test on Known Non-Stationary Time-Series

url = 'https://raw.githubusercontent.com/selva86/datasets/master/a10.csv'
df = pd.read_csv(url, parse_dates=['date'], index_col='date')
series = df.loc[:, 'value'].values
df.plot(figsize=(14,8), legend=None, title='a10 - Drug Sales Series')

# ADF Test
result = adfuller(series, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')  
    
#%%

# ADF Test on Stationary Time-Series

# ADF test on random numbers
series = np.random.randn(100)
a_list = list(range(0, 100))
result = adfuller(series, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(series);
plt.title('Random');
    
    