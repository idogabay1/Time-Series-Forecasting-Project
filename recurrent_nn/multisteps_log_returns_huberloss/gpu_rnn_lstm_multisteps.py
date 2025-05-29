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
from sklearn.metrics import mean_squared_error
from torch.nn import HuberLoss
# ----------------------------- Configuration -----------------------------
SEQ_LEN = 50
N_FORWARD = 5
EPOCHS = 1000
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TICKER = "AAPL"
EVAL_INTERVAL = 50
# ----------------------------- ReduceLROnPlateau Scheduler -----------------------------
SCHEDULER_PATIENCE = 10000
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6
# ----------------------------- Data Loader -----------------------------
def load_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    prices = df["Close"].values.reshape(-1, 1)
    
    # Step 1: scale prices
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    # Step 2: compute returns (log or simple diff)
    returns = np.diff(prices_scaled, axis=0)
    returns = np.vstack([np.zeros((1, 1)), returns])  # pad the first diff as 0

    # Step 3: build combined input: [price, return]
    full_features = np.concatenate([prices_scaled, returns], axis=1)

    X, y = [], []
    for i in range(len(full_features) - SEQ_LEN - N_FORWARD):
        X.append(full_features[i:i + SEQ_LEN])
        y.append(prices_scaled[i + SEQ_LEN : i + SEQ_LEN + N_FORWARD])  # next price(s)

    X = np.array(X)  # shape: [samples, seq_len, 2]
    y = np.array(y)  # shape: [samples, N_FORWARD, 1] (MinMax targets)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    return (
        torch.tensor(X_train, dtype=torch.float32).to(DEVICE),
        torch.tensor(y_train.squeeze(-1), dtype=torch.float32).to(DEVICE),
        torch.tensor(X_test, dtype=torch.float32).to(DEVICE),
        torch.tensor(y_test.squeeze(-1), dtype=torch.float32).to(DEVICE),
        scaler
    )

# ----------------------------- Models -----------------------------
class StockRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=2,
                          hidden_size=128, batch_first=True,
                          num_layers=3, dropout=0.2)
        self.fc = nn.Linear(128, N_FORWARD)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2,
                            hidden_size=128, batch_first=True,
                            num_layers=3, dropout=0.2)
        self.fc = nn.Linear(128, N_FORWARD)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class StockGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2,
                          hidden_size=128, batch_first=True,
                          num_layers=3, dropout=0.2)
        self.fc = nn.Linear(128, N_FORWARD)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# ----------------------------- Evaluation -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=SCHEDULER_FACTOR,
                                                     patience=SCHEDULER_PATIENCE,
                                                     min_lr=SCHEDULER_MIN_LR,threshold=1e-4,
                                                     verbose=True)
    criterion = HuberLoss(delta=1.0)#nn.MSELoss()
    best_rmse = float("inf")
    best_model_path = f"{name.lower()}_best.pt"
    initial_lr = LR
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        scheduler.step(loss.item())
        # Print if learning rate is reduced
        loss.backward()
        optimizer.step()

        if epoch % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X_test).cpu().numpy()
                targets = y_test.cpu().numpy()
                targets = targets#.squeeze(-1) 
                current_rmse = rmse(targets, preds)
            print(f"{name} Epoch {epoch}: RMSE = {current_rmse:.5f}")

            if current_rmse < best_rmse:
                best_rmse = current_rmse
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best {name} model saved (RMSE = {current_rmse:.5f})")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        targets = y_test.cpu().numpy()
    return preds, targets

# ----------------------------- Plotting -----------------------------
def plot_predictions(preds, targets, scaler, name):
    if preds.ndim == 3:
        preds = preds.squeeze(-1)  # Remove last dimension
    if targets.ndim == 3:
        targets = targets.squeeze(-1)
    preds_inv = scaler.inverse_transform(preds)
    targets_inv = scaler.inverse_transform(targets)

    plt.figure(figsize=(10, 4))
    plt.plot(targets_inv[:, 0], label="True (first step)")
    plt.plot(preds_inv[:, 0], label="Predicted (first step)")
    plt.title(f"{name} Forecast (First step only)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_forecast.png")

# ----------------------------- Main -----------------------------
def main(train_rnn=True, train_lstm=True, train_gru=True):
    X_train, y_train, X_test, y_test, scaler = load_data()
    
    if train_rnn:
        print("\nTraining RNN...")
        rnn = StockRNN()
        preds, targets = train_model(rnn, X_train, y_train, X_test, y_test, name="RNN")
        plot_predictions(preds, targets, scaler, "RNN")

    if train_lstm:
        print("\nTraining LSTM...")
        lstm = StockLSTM()
        preds, targets = train_model(lstm, X_train, y_train, X_test, y_test, name="LSTM")
        plot_predictions(preds, targets, scaler, "LSTM")

    if train_gru:
        print("\nTraining GRU...")
        gru = StockGRU()
        preds, targets = train_model(gru, X_train, y_train, X_test, y_test, name="GRU")
        plot_predictions(preds, targets, scaler, "GRU")

if __name__ == "__main__":
    main(train_rnn=True, train_lstm=True, train_gru=True)
