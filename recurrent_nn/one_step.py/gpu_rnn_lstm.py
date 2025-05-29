import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Hyperparameters ---
SEQ_LEN = 50
EPOCHS = 1000
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load and preprocess data ---
def load_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(len(prices_scaled) - SEQ_LEN):
        X.append(prices_scaled[i:i + SEQ_LEN])
        y.append(prices_scaled[i + SEQ_LEN])

    X = np.array(X)
    y = np.array(y)

    # Split into train/test
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    return (torch.tensor(X_train, dtype=torch.float32).to(DEVICE),
            torch.tensor(y_train, dtype=torch.float32).to(DEVICE),
            torch.tensor(X_test, dtype=torch.float32).to(DEVICE),
            torch.tensor(y_test, dtype=torch.float32).to(DEVICE),
            scaler)

# --- Models ---
class StockRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1,
                    hidden_size=64, batch_first=True,
                    num_layers=3, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=64, batch_first=True,
                            num_layers=3, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class StockGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1,
                          hidden_size=64, batch_first=True,
                          num_layers=3, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# --- Training ---
def train_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)  # Removed .unsqueeze(-1)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 5 == 0:
            print(f"{name} Epoch {epoch}: Loss = {loss.item():.5f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()  # Removed .unsqueeze(-1)
        targets = y_test.cpu().numpy()

    return preds, targets, train_losses


# --- Main ---
def main(train_rnn=True, train_lstm=True, train_gru=True):
    X_train, y_train, X_test, y_test, scaler = load_data()

    if train_rnn:
        print("\nTraining RNN...")
        rnn_model = StockRNN()
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
        rnn_preds, rnn_targets, rnn_loss = train_model(rnn_model, X_train, y_train, X_test, y_test, name="RNN")
        plot_results(rnn_preds, rnn_targets, scaler, "RNN")

    if train_lstm:
        print("\nTraining LSTM...")
        lstm_model = StockLSTM()
        lstm_preds, lstm_targets, lstm_loss = train_model(lstm_model, X_train, y_train, X_test, y_test, name="LSTM")
        plot_results(lstm_preds, lstm_targets, scaler, "LSTM")

    if train_gru:
        print("\nTraining GRU...")
        gru_model = StockGRU()
        gru_preds, gru_targets, gru_loss = train_model(gru_model, X_train, y_train, X_test, y_test, name="GRU")
        plot_results(gru_preds, gru_targets, scaler, "GRU")

# --- Plotting ---
def plot_results(preds, targets, scaler, title):
    preds_rescaled = scaler.inverse_transform(preds)
    targets_rescaled = scaler.inverse_transform(targets)
    plt.figure(figsize=(10, 4))
    plt.plot(targets_rescaled, label="True")
    plt.plot(preds_rescaled, label="Predicted")
    plt.title(f"{title} Prediction")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{title.lower()}_prediction.png")

if __name__ == "__main__":
    main(train_rnn=True, train_lstm=True, train_gru=True)
