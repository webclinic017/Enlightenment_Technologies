from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# From https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/

# Augmented Dickey Fuller test (ADF Test) is a common statistical test used to test 
# whether a given Time series is stationary or not. It is one of the most commonly 
# used statistical test when it comes to analyzing the stationary of a series.

# the ADF test is fundamentally a statistical significance test. That means, there 
# is a hypothesis testing involved with a null and alternate hypothesis and as a 
# result a test statistic is computed and p-values get reported. It is from the test 
# statistic and the p-value, you can make an inference as to whether a given series is stationary or not.

# A key point to remember here is: Since the null hypothesis assumes the presence of unit 
# root, that is α=1, the p-value obtained should be less than the significance level (
# say 0.05) in order to reject the null hypothesis. Thereby, inferring that the series 
# is stationary. However, this is a very common mistake analysts commit with this test. 
# That is, if the p-value is less than significance level, people mistakenly take the 
# series to be non-stationary.



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
    
    