# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the data
data = pd.read_csv('fdi_inflow.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index(['Year'], inplace=True)

# Plot the data
data.plot()
plt.ylabel('FDI Inflow (Thousands of Dollars)')
plt.xlabel('Year')
plt.show()

# ARIMA
q = d = range(0, 2)
p = range(0, 4)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 1) for x in list(itertools.product(p, d, q))]

AIC = []
SARIMAX_model = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=0)
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

best_model_index = np.argmin(AIC)
best_model_params = SARIMAX_model[best_model_index]

print(f'The best model has AIC={AIC[best_model_index]} and parameters: ARIMA{best_model_params[0]} x{best_model_params[1]}')

# Fit the best model
mod = sm.tsa.statespace.SARIMAX(data,
                                order=best_model_params[0],
                                seasonal_order=best_model_params[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=0)

# Forecast
forecast_steps = 5  # You can adjust this based on how many steps into the future you want to forecast
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='A')[1:]

# Plot the results
ax = data.plot(figsize=(12, 6))
forecast.predicted_mean.plot(ax=ax, label=f'Forecast ({forecast_steps} steps ahead)')
ax.fill_between(forecast_index, forecast.conf_int()['lower FDI Inflow'], forecast.conf_int()['upper FDI Inflow'], color='k', alpha=.1)
plt.ylabel('FDI Inflow (Thousands of Dollars)')
plt.xlabel('Year')
plt.legend()
plt.show()
