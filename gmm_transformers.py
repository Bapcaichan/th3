import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import chrono
import asyncio
import platform
from io import StringIO


# Fixed RobustStandardScaler
class RobustStandardScaler(StandardScaler):
    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float32)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        if np.any(np.isnan(self.mean_)) or np.any(np.isnan(self.scale_)):
            print("Warning: NaN in scaler mean or scale. Filling with 0 for mean and 1 for scale.")
            self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
            self.scale_ = np.nan_to_num(self.scale_, nan=1.0)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        X_transformed = (X - self.mean_) / self.scale_
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        return np.clip(X_transformed, -5, 5)

    def transform(self, X):
        X = np.array(X, dtype=np.float32)
        X_transformed = (X - self.mean_) / self.scale_
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        return np.clip(X_transformed, -5, 5)

    def inverse_transform(self, X):
        X = np.array(X, dtype=np.float32)
        return X * self.scale_ + self.mean_


# Autoformer classes (unchanged)
class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_avg = nn.functional.avg_pool1d(x_padded.permute(0, 2, 1), kernel_size=self.kernel_size, stride=self.stride,
                                         padding=0)
        return x_avg.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        Q = self.query_projection(queries)
        K = self.key_projection(keys)
        V = self.value_projection(values)
        score = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_model ** 0.5)
        attn = nn.functional.softmax(score + 1e-8, dim=-1)
        out = torch.matmul(attn, V)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, kernel_size, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = AutoCorrelationLayer(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.auto_corr(seasonal, seasonal, seasonal)
        seasonal = self.dropout(self.norm(seasonal + x))
        out = self.ff(seasonal)
        return self.decomp(out + seasonal)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, kernel_size, dropout=0.1):
        super().__init__()
        self.decomp1 = SeriesDecomposition(kernel_size)
        self.auto_corr1 = AutoCorrelationLayer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.decomp2 = SeriesDecomposition(kernel_size)
        self.auto_corr2 = AutoCorrelationLayer(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, memory):
        seasonal, trend = self.decomp1(x)
        seasonal = self.auto_corr1(seasonal, seasonal, seasonal)
        seasonal = self.dropout1(self.norm1(seasonal + x))
        seasonal2, trend2 = self.decomp2(seasonal)
        seasonal2 = self.auto_corr2(seasonal2, memory, memory)
        seasonal2 = self.dropout2(self.norm2(seasonal2 + seasonal))
        out = self.ff(seasonal2)
        return self.decomp2(out + seasonal2)


class Autoformer(nn.Module):
    def __init__(self, input_dim, d_model, seq_len, pred_len, enc_layers=2, dec_layers=1, kernel_size=25):
        super().__init__()
        self.enc_input = nn.Linear(input_dim, d_model)
        self.dec_input = nn.Linear(input_dim, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, kernel_size) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, kernel_size) for _ in range(dec_layers)])
        self.projection = nn.Linear(d_model, 1)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x_enc, x_dec):
        enc_out = self.enc_input(x_enc)
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)

        dec_out = self.dec_input(x_dec)
        for layer in self.decoder:
            dec_out, _ = layer(dec_out, enc_out)

        out = self.projection(dec_out)
        return out.squeeze(-1)


# Helper function to parse CSV data
def parse_csv_data(csv_string):
    try:
        df = pd.read_csv(StringIO(csv_string), parse_dates=['time'])
        features = ['total load actual', 'total load forecast', 'price actual', 'generation solar',
                    'generation wind onshore']
        if not all(col in df.columns for col in features):
            missing_cols = [col for col in features if col not in df.columns]
            raise ValueError(f"Required columns missing: {', '.join(missing_cols)}")

        df = df[features]

        print("Initial data stats:\n", df.describe())
        print("NaN counts per column:\n", df.isna().sum())
        print("Data types:\n", df.dtypes)

        df = df.fillna(method='ffill')
        df = df.fillna(df.median(numeric_only=True))

        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                raise ValueError(f"Column {col} contains all NaNs after coercion")
            df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
            if df[col].std() == 0 or np.isnan(df[col].std()):
                print(f"Warning: Column {col} has zero variance or NaN std. Adding small noise.")
                df[col] += np.random.normal(0, 1e-6, size=len(df))

        nan_rows = df.isna().any(axis=1).sum()
        if nan_rows > 0:
            print(f"Warning: Dropping {nan_rows} rows with NaNs")
            df = df.dropna()

        if df.isna().any().any():
            raise ValueError("NaNs detected after preprocessing")

        print("Processed data stats:\n", df.describe())
        return df
    except Exception as e:
        print(f"Error parsing CSV data: {e}")
        return None


# Prepare sequences for Autoformer
def prepare_sequences(df, seq_len, pred_len, feature_cols, target_col):
    try:
        if len(df) < seq_len + pred_len:
            raise ValueError(f"Dataset too small: {len(df)} rows, need at least {seq_len + pred_len}")

        # Check raw data for NaNs
        if df[feature_cols].isna().any().any():
            print("NaNs detected in raw feature data")
        if df[target_col].isna().any():
            print("NaNs detected in raw target data")

        scaler = RobustStandardScaler()
        data = scaler.fit_transform(df[feature_cols])
        print("Feature scaler mean:", scaler.mean_)
        print("Feature scaler scale:", scaler.scale_)
        if np.any(np.isnan(data)):
            print("NaNs detected in scaled feature data")
            raise ValueError("NaNs in scaled feature data")

        target_scaler = RobustStandardScaler()
        target = target_scaler.fit_transform(df[[target_col]])
        print("Target scaler mean:", target_scaler.mean_)
        print("Target scaler scale:", target_scaler.scale_)
        if np.any(np.isnan(target)):
            print("NaNs detected in scaled target data")
            raise ValueError("NaNs in scaled target data")

        x_enc, x_dec, y = [], [], []
        total_len = seq_len + pred_len

        for i in range(len(df) - total_len + 1):
            x_enc.append(data[i:i + seq_len])
            x_dec.append(data[i + seq_len:i + total_len])
            y.append(target[i + seq_len:i + total_len])

        x_enc = np.array(x_enc)
        x_dec = np.array(x_dec)
        y = np.array(y)

        if np.any(np.isnan(x_enc)):
            print("NaNs detected in x_enc")
        if np.any(np.isnan(x_dec)):
            print("NaNs detected in x_dec")
        if np.any(np.isnan(y)):
            print("NaNs detected in y")

        if np.any(np.isnan(x_enc)) or np.any(np.isnan(x_dec)) or np.any(np.isnan(y)):
            raise ValueError("NaNs detected in prepared sequences")

        train_size = int(0.8 * len(x_enc))
        if train_size == 0:
            raise ValueError("Not enough sequences for training after splitting")
        x_enc_train, x_enc_val = x_enc[:train_size], x_enc[train_size:]
        x_dec_train, x_dec_val = x_dec[:train_size], x_dec[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        return (torch.tensor(x_enc_train, dtype=torch.float32),
                torch.tensor(x_dec_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(x_enc_val, dtype=torch.float32),
                torch.tensor(x_dec_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                target_scaler)
    except Exception as e:
        print(f"Error preparing sequences: {e}")
        return None


# Training function
def train_autoformer(model, x_enc, x_dec, y, epochs=10, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    dataset_size = x_enc.shape[0]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for i in range(0, dataset_size, batch_size):
            end = min(i + batch_size, dataset_size)
            batch_x_enc = x_enc[i:end]
            batch_x_dec = x_dec[i:end]
            batch_y = y[i:end]

            optimizer.zero_grad()
            output = model(batch_x_enc, batch_x_dec)
            if torch.isnan(output).any():
                print(f"NaN detected in model output at epoch {epoch + 1}, batch {i // batch_size + 1}")
                return None
            loss = criterion(output, batch_y.squeeze(-1))
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch + 1}, batch {i // batch_size + 1}")
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / batch_count:.4f}")
    return model


# Predict with GMM adjustment
def predict_with_gmm(model, x_enc, x_dec, x_enc_val, x_dec_val, y_val, target_scaler, gmm):
    model.eval()
    with torch.no_grad():
        val_pred = model(x_enc_val, x_dec_val).numpy()
        val_actual = y_val.squeeze(-1).numpy()
        residuals = target_scaler.inverse_transform(val_pred) - target_scaler.inverse_transform(val_actual)

        gmm.fit(residuals.reshape(-1, 1))

        pred = model(x_enc, x_dec).numpy()
        pred = target_scaler.inverse_transform(pred)

        adjustments, _ = gmm.sample(pred.shape[0] * pred.shape[1])
        adjustments = adjustments.reshape(pred.shape)

        pred_adjusted = pred + adjustments
        return pred_adjusted


# Main execution
def main(csv_data=None):
    if csv_data is None:
        try:
            df = pd.read_csv("energy_dataset.csv")
        except FileNotFoundError:
            print("Error: energy_dataset.csv not found. Please provide CSV data or use loadFileData in Pyodide.")
            return None
    else:
        df = parse_csv_data(csv_data)
        if df is None:
            print("Error: Failed to parse provided CSV data.")
            return None

    seq_len = 96
    pred_len = 24
    feature_cols = ['total load actual', 'total load forecast', 'price actual', 'generation solar',
                    'generation wind onshore']
    input_dim = len(feature_cols)
    d_model = 64
    enc_layers = 2
    dec_layers = 1
    kernel_size = 25
    target_col = 'total load actual'

    data = prepare_sequences(df, seq_len, pred_len, feature_cols, target_col)
    if data is None:
        print("Error: Failed to prepare sequences.")
        return None

    x_enc_train, x_dec_train, y_train, x_enc_val, x_dec_val, y_val, target_scaler = data

    model = Autoformer(input_dim=input_dim, d_model=d_model, seq_len=seq_len, pred_len=pred_len,
                       enc_layers=enc_layers, dec_layers=dec_layers, kernel_size=kernel_size)
    model = train_autoformer(model, x_enc_train, x_dec_train, y_train)
    if model is None:
        print("Error: Training failed due to NaN values.")
        return None

    gmm = GaussianMixture(n_components=3, random_state=42)

    last_x_enc = x_enc_train[-1:]
    last_x_dec = x_dec_train[-1:]
    pred = predict_with_gmm(model, last_x_enc, last_x_dec, x_enc_val, x_dec_val, y_val, target_scaler, gmm)

    actual = target_scaler.inverse_transform(y_train[-1].squeeze(-1).numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(range(pred_len), actual, label='Actual Load', color='blue')
    plt.plot(range(pred_len), pred[0], label='Predicted Load', color='red', linestyle='--')
    plt.title('Total Load Actual vs Predicted')
    plt.xlabel('Hour Ahead')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('load_prediction.png')

    return pred[0]


# Run in Pyodide-compatible async mode
async def async_main():
    try:
        csv_data = None  # Replace with: csv_data = loadFileData("energy_dataset.csv")
        pred = main(csv_data)
        if pred is not None:
            print("Predictions for next 24 hours:", pred)
        return pred
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if platform.system() == "Emscripten":
    asyncio.ensure_future(async_main())
else:
    if __name__ == "__main__":
        asyncio.run(async_main())