
# Time Series Forecasting Project

This project demonstrates a variety of methods for time series forecasting applied to stock prices and airline passenger data. It includes classic statistical models as well as modern deep learning approaches. The aim is to compare the performance and interpretability of each method.

---

## 🔍 Project Structure

- **recurrent_nn/**: Deep learning models using RNN, GRU, and LSTM
  - `multisteps_log_returns_huberloss/`: Multistep forecasting with log-returns and Huber loss
  - `one_step.py/`: One-step forecasting with baseline models
- **attention/**: Transformer and Seq2Seq models with attention mechanisms
- **kalman/**: Kalman Filter implementation for smoothed predictions
- **sarima/**: SARIMA and ARIMA models using statsmodels
- **exponential smoothing (ets)/**: Forecasting using ETS

---

## 🧠 Models Used

| Model Type           | Framework         | Description |
|----------------------|-------------------|-------------|
| RNN / LSTM / GRU     | PyTorch           | Sequence models for time-series forecasting |
| Seq2Seq + Attention  | PyTorch           | Encoder-decoder model with learned attention weights |
| Transformer          | PyTorch           | Self-attention-based model |
| SARIMA / ARIMA       | statsmodels       | Classical statistical forecasting |
| ETS (Exponential Smoothing) | statsmodels | Trend and seasonality modeling |
| Kalman Filter        | Numpy             | Recursive Bayesian estimation technique |

---

## 📊 Sample Results

Below are sample visualizations from the prediction models:

### Forecast Comparisons (RNN, GRU, LSTM)
![RNN Forecast](/recurrent_nn/multisteps_log_returns_huberloss/n_5/rnn_forecast.png)
![GRU Forecast](/recurrent_nn/multisteps_log_returns_huberloss/n_5/gru_forecast.png)
![LSTM Forecast](/recurrent_nn/multisteps_log_returns_huberloss/n_5/lstm_forecast.png)

### Transformer Step-by-Step Prediction
![Transformer Step 10](/attention/GE_step10_prediction.png)
![Transformer Full Series](/attention/GE_full_series_prediction.png)

### Kalman Filter Output
![Kalman Filtered Price](/kalman/kalman_filtered_price.png)

### ETS Forecast
![ETS_Forecast](/exponential_smoothing_(ets)/ets_forecast.png)

---

## 📁 How to Run

This project is organized by method. To run a specific model:

1. Install dependencies: `pip install -r requirements.txt`
2. Navigate to the relevant folder.
3. Run the script, e.g.:
   ```bash
   python attention/main.py
   ```

---

## ✍️ Author

Project by [Ido Gabay](https://github.com/idogabay1)

---

## 📄 License

MIT License
