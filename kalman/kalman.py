import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import yfinance as yf

data = yf.download("AAPL", start="2022-01-01", end="2024-01-01")
prices = data["Close"].dropna().astype(float)
dates = prices.index

d = 2
if d == 1:
    print("Using 1D Kalman Filter")
    # 1d
    transition_matrices = [1] 
    observation_matrices = [1]
    initial_state_mean = np.array([prices.iloc[0].item()])
    initial_state_covariance = 0.5
    observation_covariance = 3
    transition_covariance = 0.5
elif d == 2:
    print("Using 2D Kalman Filter")
# 2d
    transition_matrices = np.array([[1, 1], [0, 1]])
    observation_matrices = np.array([[1, 0]])
    initial_state_mean = np.array([prices.iloc[0].item(), 0])
    initial_state_covariance = np.array([[1, 0], [0, 1]])
    observation_covariance = np.array([[1]])  # Adjusted to a scalar for 2D
    transition_covariance = np.array([[0.01, 0], [0, 0.01]])


kf = KalmanFilter(
    transition_matrices=transition_matrices,
    observation_matrices=observation_matrices,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance
)

state_means, _ = kf.filter(prices.values)
if d == 2:
    estimated_prices = state_means[:, 0]
    estimated_velocity = state_means[:, 1]
    
# Plot estimated price vs real price
    plt.figure(figsize=(12, 5))
    plt.plot(dates, prices, label='Observed Price', alpha=0.5)
    plt.plot(dates, estimated_prices, label='Kalman Estimated Price', linewidth=2)
    plt.title("Kalman Filter (2D) – Price and Trend Estimation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("kalman_2d_price.png")

    # Optional: Plot velocity (price trend)
    plt.figure(figsize=(12, 3))
    plt.plot(dates, estimated_velocity, label='Estimated Velocity (Price Change)', color='orange')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title("Estimated Trend (Velocity) Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("kalman_2d_velocity.png")
    # print emse
    mse = mean_squared_error(prices, estimated_prices)
    print(f"Mean Squared Error: {mse:.2f}")
else:

    filtered = pd.Series(state_means.flatten(), index=prices.index)
    plt.figure(figsize=(12, 5))
    prices.plot(label="Original", alpha=0.6)
    filtered.plot(label="Kalman Filter", linewidth=2)
    plt.title("Kalman Filter – Smoothed Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"kalman_filtered_price_parameters_{transition_matrices[0]}_{observation_matrices[0]}_{initial_state_covariance}_{observation_covariance}_{transition_covariance}.png")

    # print emse
    mse = mean_squared_error(prices, filtered)
    print(f"Mean Squared Error: {mse:.2f}")