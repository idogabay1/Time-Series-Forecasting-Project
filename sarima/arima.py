import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load example dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Plot the time series
df.plot(title='Monthly Air Passengers')
# plt.show()
# save the plot to a file
plt.savefig('airline_passengers.png')

# Check stationarity (simple diff)
diff = df.diff().dropna()
diff.plot(title='Differenced Series')
# plt.show()
# save the differenced plot to a file
plt.savefig('differenced_series.png')

# Fit ARIMA model
model_arima = ARIMA(df, order=(4,1,4))  # p=2, d=1, q=2
model_arima_fit = model_arima.fit()

# Forecast next 12 months
forecast_arima = model_arima_fit.forecast(steps=224)
df_forecast = pd.concat([df, forecast_arima.rename('Forecast')], axis=1)

# Plot
df_forecast.plot(title='ARIMA Forecast')
# plt.show()
# save the ARIMA forecast plot to a file
plt.savefig('arima_forecast.png')




# Fit SARIMA model (seasonal order for monthly data: S=12)
model_sarima = SARIMAX(df, 
                       order=(2,1,2),           # ARIMA part: p,d,q
                       seasonal_order=(2,1,2,12))  # Seasonal part: P,D,Q,S

model_sarima_fit = model_sarima.fit()

# Forecast 12 steps ahead
forecast_sarima = model_sarima_fit.forecast(steps=224)
df_forecast_sarima = pd.concat([df, forecast_sarima.rename('SARIMA Forecast')], axis=1)

# Plot
df_forecast_sarima.plot(title='SARIMA Forecast')
#plt.show()
# save the SARIMA forecast plot to a file
plt.savefig('sarima_forecast.png')

