
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pickle

# --- Hyperparameters ---
SEQ_LEN = 300
N_FORWARD = 10
EPOCHS = 102
LR = 0.001
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Sinusoidal Positional Encoding ---
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: [1, seq_len, d_model]

# --- Load data from multiple tickers ---
def load_data(tickers=None, start="2004-01-01", end="2024-01-01"):
    cache_path = "cached_data.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            X, y, scaler = pickle.load(f)
    else:
        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
                       "HD", "MA", "DIS", "PEP", "BAC", "KO", "PFE", "NKE", "ADBE", "CSCO",
                       "INTC", "T", "WMT"]
            """, "CVX", "ABT", "CRM", "XOM", "MRK", "MCD", "LLY",
                       "AVGO", "CMCSA", "ACN", "COST", "MDT", "DHR", "TXN", "NEE", "LIN", "ORCL",
                       "WFC", "UPS", "BMY", "PM", "IBM", "UNP", "QCOM", "RTX", "AMGN", "GILD",
                       "SBUX", "ISRG", "LMT", "NOW", "TMO", "INTU", "BLK", "ZTS", "LOW", "BA",
                       "GE", "CAT", "SPGI", "AMT", "PLD", "MO", "CI", "AXP", "MS", "SCHW",
                       "ADP", "CB", "SYK", "GM", "BKNG", "EL", "C", "USB", "SO", "DE",
                       "ADI", "MMC", "MDLZ", "CL", "FDX", "GD", "REGN", "APD", "ECL", "BDX",
                       "MAR", "DUK", "SHW", "PSA", "EXC", "AON", "CSX", "NSC", "HUM", "ITW"]"""

        all_X, all_y = [], []
        scaler = MinMaxScaler()
        for ticker in tqdm(tickers[:100], desc="Downloading data"):
            try:
                df = yf.download(ticker, start=start, end=end)
                if df.empty or 'Close' not in df: continue
                prices = df["Close"].values.reshape(-1, 1)
                prices_scaled = scaler.fit_transform(prices)
                returns = np.diff(prices_scaled, axis=0)
                returns = np.vstack([np.zeros((1, 1)), returns])
                features = np.concatenate([prices_scaled, returns], axis=1)

                for i in range(len(features) - SEQ_LEN - N_FORWARD):
                    X = features[i:i + SEQ_LEN]
                    y = prices_scaled[i + SEQ_LEN : i + SEQ_LEN + N_FORWARD]

                    all_X.append(X)
                    all_y.append(y)
            except Exception:
                continue

        X = np.array(all_X)
        y = np.array(all_y).reshape(-1, N_FORWARD)
        with open(cache_path, "wb") as f:
            pickle.dump((X, y, scaler), f)

    split = int(0.8 * len(X))

    train_ds = TensorDataset(torch.tensor(X[:split], dtype=torch.float32),
                             torch.tensor(y[:split], dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X[split:], dtype=torch.float32),
                            torch.tensor(y[split:], dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, SEQ_LEN)
        self.v = nn.Parameter(torch.rand(SEQ_LEN))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        hidden = hidden.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attn_weights = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attn_weights, dim=1)

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, N_FORWARD)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(DEVICE)
        encoder_outputs, hidden = self.encoder(x)
        encoder_outputs = self.dropout(encoder_outputs)
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        _, decoder_hidden = self.decoder(context)
        return self.fc(decoder_hidden[-1])

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=2, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = positional_encoding(SEQ_LEN, d_model).to(DEVICE)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, N_FORWARD)

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        return self.fc(x[:, -1])

# Training, Plotting and Main remain unchanged...
def train_model(model, train_loader, test_loader, scaler, name="Model"):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='min', factor=0.8, patience=200, verbose=True)
    criterion = nn.HuberLoss()
    best_rmse = float("inf")
    best_model_path = f"{name.lower()}_best.pt"
    patience, wait = 200, 0
    # add progress bar
    # for epoch in range(EPOCHS):
    for epoch in tqdm(range(EPOCHS), desc=f"Training {name}"):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            # print("batch_y shape:", batch_y.shape)

            output = model(batch_x).squeeze()
            # print("output shape:", output.shape)
            loss = criterion(output, batch_y)
            loss.backward()
            scheduler.step(loss)
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x).squeeze()
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        current_rmse = np.sqrt(mean_squared_error(targets, preds))

        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"{name} Epoch {epoch}: RMSE = {current_rmse:.5f}")

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            torch.save(model.state_dict(), best_model_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\u23f9\ufe0f Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        final_preds, final_targets = [], []
        for batch_x, batch_y in test_loader:
            output = model(batch_x).squeeze()
            final_preds.append(output.cpu().numpy())
            final_targets.append(batch_y.cpu().numpy())

    return np.concatenate(final_preds), np.concatenate(final_targets)



# --- Plotting ---
def plot_predictions(preds, targets, scaler, name):
    preds = preds.reshape(-1, 1)
    targets = targets.reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    targets_inv = scaler.inverse_transform(targets)

    plt.figure(figsize=(10, 4))
    plt.plot(targets_inv, label="True")
    plt.plot(preds_inv, label="Predicted")
    plt.title(f"{name} Prediction")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_prediction.png")







# prediction_plot.py
# Load a trained model, pick a stock, and visualize the prediction vs real data

# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import yfinance as yf
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# from model import TransformerTimeSeries  # assumes your model class is in model.py

# --- Settings ---
SEQ_LEN = 300
N_FORWARD = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "attention/transformer_best_n_10.pt"
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# --- Load stock and prepare data ---
def prepare_input_series(ticker: str, start=START_DATE, end=END_DATE):
    df = yf.download(ticker, start=start, end=end)
    prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    returns = np.diff(scaled, axis=0)
    returns = np.vstack([np.zeros((1, 1)), returns])
    features = np.concatenate([scaled, returns], axis=1)

    X = []
    for i in range(len(features) - SEQ_LEN - N_FORWARD):
        X.append(features[i:i + SEQ_LEN])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    target_indices = range(SEQ_LEN, SEQ_LEN + len(X))
    return X_tensor, scaler, prices, df.index, target_indices

# --- Predict and plot specific step X ---
def predict_and_plot_step(ticker: str, step: int):
    assert 1 <= step <= N_FORWARD, f"Step must be between 1 and {N_FORWARD}"

    model = TransformerTimeSeries(input_dim=2, d_model=32, nhead=4, num_layers=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    X_tensor, scaler, original_prices, date_index, target_indices = prepare_input_series(ticker)
    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()  # shape: [samples, N_FORWARD]

    # Extract predictions for specific step
    preds_step_scaled = preds_scaled[:, step - 1]
    preds_price = scaler.inverse_transform(preds_step_scaled.reshape(-1, 1)).flatten()

    # Align ground truth
    true_prices = original_prices[SEQ_LEN + step - 1:SEQ_LEN + step - 1 + len(preds_price)].flatten()
    plot_dates = date_index[SEQ_LEN + step - 1:SEQ_LEN + step - 1 + len(preds_price)]

    # Calculate average percentage error
    perc_error = np.abs((preds_price - true_prices) / true_prices) * 100
    avg_perc_error = np.mean(perc_error)
    print(f"Average % Error for step {step}: {avg_perc_error:.2f}%")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(plot_dates, true_prices, label="True Prices")
    plt.plot(plot_dates, preds_price, label=f"Predicted Prices (step {step})")
    plt.title(f"{ticker} - Prediction for Step {step}\nAvg % Error: {avg_perc_error:.2f}%")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{ticker}_step{step}_prediction.png")
    print(f"Prediction for step {step} saved to {ticker}_step{step}_prediction.png")
    return avg_perc_error


# --- Main ---
def main(train_seq2seq=False, train_transformer=True):
    # X_train, y_train, X_test, y_test, scaler = load_data()
    train = False
    TICKER = "GE"  # Default ticker for prediction
    if train:
            
        train_loader, test_loader, scaler = load_data()


        if train_seq2seq:
            print("\nTraining Seq2Seq with Attention...")
            seq_model = Seq2SeqWithAttention(input_dim=2, hidden_dim=128,
                                            dropout=0.2)
            seq_model.to(DEVICE)
            seq_model.train()
            preds, targets = train_model(seq_model, train_loader, test_loader, scaler, name="Seq2Seq")
            plot_predictions(preds, targets, scaler, "Seq2Seq")

        if train_transformer:
            print("\nTraining Transformer...")
            trans_model = TransformerTimeSeries(input_dim=2, d_model=32,
                                                nhead=4, num_layers=2,dropout=0.2)
            trans_model.to(DEVICE)
            trans_model.train()
            preds, targets = train_model(trans_model, train_loader, test_loader, scaler, name="Transformer")
            plot_predictions(preds, targets, scaler, "Transformer")
    else:
        errs = []
        for step in range(1, N_FORWARD):
            err = predict_and_plot_step(TICKER, step=step)  # Change step as needed
            errs.append(err)
        # save plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, N_FORWARD), errs, marker='o')
        plt.title(f"Average Percentage Error for {TICKER} Predictions")
        plt.xlabel("Prediction Step")
        plt.ylabel("Average % Error")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{TICKER}_avg_perc_error.png")

if __name__ == "__main__":
    main()
