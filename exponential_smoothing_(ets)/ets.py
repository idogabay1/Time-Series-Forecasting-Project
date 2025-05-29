import matplotlib
matplotlib.use('Agg')  # פתרון ל־WSL, משתמש ב־backend ללא GUI
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#import sarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# נטען את הדאטה – סדרה עם עונתיות ברורה
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
y = df['Passengers']

# נבנה מודל ETS עם מגמה ועונתיות (additive)
model = ExponentialSmoothing(y,
                              trend='add',
                              seasonal='multiplicative',
                              seasonal_periods=11)  # עונתיות של שנה

fit = model.fit()

# חיזוי קדימה
forecast = fit.forecast(steps=20)

# הצגה
y.plot(label='Actual', figsize=(10, 5))
forecast.plot(label='Forecast', color='red')
plt.title("ETS (Additive) Forecast")
plt.legend()
plt.grid()
plt.tight_layout()
# plt.show()
# save the plot to a file
plt.savefig('ets_forecast.png')
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = y[-20:]  # נניח שהחיזוי הוא על 224 התצפיות האחרונות
y_pred = forecast[-20:]  # החיזוי שלנו
# חישוב RMSE    


rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f'RMSE: {rmse:.2f}')

model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
print(f"AIC: {model.aic}")
print(f"BIC: {model.bic}")