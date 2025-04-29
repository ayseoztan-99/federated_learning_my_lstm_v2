import flwr as fl
import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from config import SERVER_ADDRESS, CLIENT_DATA_DIR, TH, TD, TW, TP, LOCAL_EPOCHS
from model import build_multi_lstm_model

import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"[Client {client_id}] Initializing...")

file_path = os.path.join(CLIENT_DATA_DIR, f"client_data_{client_id}.csv")
df = pd.read_csv(file_path)
df = df.sort_values(by=["location", "timestep"]).reset_index(drop=True)
locations = df["location"].unique()

timesteps_per_day = 288
train_days = 50
test_days = 12
train_size = train_days * timesteps_per_day
test_size = test_days * timesteps_per_day

# --------- TRAIN/TEST HAM AYIRIMI ---------
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:train_size + test_size]

# --------- TRAIN SEQUENCE OLUŞTUR ---------
X_recent_all, X_daily_all, X_weekly_all, Y_all = [], [], [], []
for loc in locations:
    df_loc = train_data[train_data["location"] == loc].reset_index(drop=True)
    for i in range(max(TH, TD, TW), len(df_loc) - TP):
        recent = df_loc["flow"].iloc[i - TH:i].values
        daily = df_loc["flow"].iloc[i - TD - timesteps_per_day:i - timesteps_per_day].values
        weekly = df_loc["flow"].iloc[i - TW - 7 * timesteps_per_day:i - 7 * timesteps_per_day].values
        target = df_loc["flow"].iloc[i:i + TP].values
        if len(recent) == TH and len(daily) == TD and len(weekly) == TW and len(target) == TP:
            X_recent_all.append(recent)
            X_daily_all.append(daily)
            X_weekly_all.append(weekly)
            Y_all.append(target)
X_recent_all = np.array(X_recent_all).reshape(-1, TH, 1)
X_daily_all = np.array(X_daily_all).reshape(-1, TD, 1)
X_weekly_all = np.array(X_weekly_all).reshape(-1, TW, 1)
Y_all = np.array(Y_all).reshape(-1, TP)

# --------- %80 TRAIN, %20 VAL BÖLÜNMESİ ---------
num_total = X_recent_all.shape[0]
num_train = int(num_total * 0.8)

X_recent_train = X_recent_all[:num_train]
X_daily_train = X_daily_all[:num_train]
X_weekly_train = X_weekly_all[:num_train]
Y_train = Y_all[:num_train]

X_recent_val = X_recent_all[num_train:]
X_daily_val = X_daily_all[num_train:]
X_weekly_val = X_weekly_all[num_train:]
Y_val = Y_all[num_train:]

# --------- TEST SEQUENCE OLUŞTUR ---------
X_test_recent, X_test_daily, X_test_weekly, Y_test = [], [], [], []
for loc in locations:
    df_loc = test_data[test_data["location"] == loc].reset_index(drop=True)
    for i in range(max(TH, TD, TW), len(df_loc) - TP):
        recent = df_loc["flow"].iloc[i - TH:i].values
        daily = df_loc["flow"].iloc[i - TD - timesteps_per_day:i - timesteps_per_day].values
        weekly = df_loc["flow"].iloc[i - TW - 7 * timesteps_per_day:i - 7 * timesteps_per_day].values
        target = df_loc["flow"].iloc[i:i + TP].values
        if len(recent) == TH and len(daily) == TD and len(weekly) == TW and len(target) == TP:
            X_test_recent.append(recent)
            X_test_daily.append(daily)
            X_test_weekly.append(weekly)
            Y_test.append(target)
X_test_recent = np.array(X_test_recent).reshape(-1, TH, 1)
X_test_daily = np.array(X_test_daily).reshape(-1, TD, 1)
X_test_weekly = np.array(X_test_weekly).reshape(-1, TW, 1)
Y_test = np.array(Y_test).reshape(-1, TP)

# --------- SCALER (sadece TRAIN ile fit) ---------
scaler_y = MinMaxScaler()
scaler_y.fit(Y_train)
Y_train_scaled = scaler_y.transform(Y_train)
Y_val_scaled = scaler_y.transform(Y_val)
Y_test_scaled = scaler_y.transform(Y_test)

# --------- MODEL ---------
model = build_multi_lstm_model(TH, TD, TW, TP)

# --------- MANUEL GET/SET WEIGHTS ---------
def get_model_weights(model):
    weights = []
    # Recent
    weights.extend([
        model.recent_model.lstm.Wf, model.recent_model.lstm.Wi, model.recent_model.lstm.Wc, model.recent_model.lstm.Wo,
        model.recent_model.lstm.bf, model.recent_model.lstm.bi, model.recent_model.lstm.bc, model.recent_model.lstm.bo,
        model.recent_model.dense.W, model.recent_model.dense.b,
    ])
    # Daily
    weights.extend([
        model.daily_model.lstm.Wf, model.daily_model.lstm.Wi, model.daily_model.lstm.Wc, model.daily_model.lstm.Wo,
        model.daily_model.lstm.bf, model.daily_model.lstm.bi, model.daily_model.lstm.bc, model.daily_model.lstm.bo,
        model.daily_model.dense.W, model.daily_model.dense.b,
    ])
    # Weekly
    weights.extend([
        model.weekly_model.lstm.Wf, model.weekly_model.lstm.Wi, model.weekly_model.lstm.Wc, model.weekly_model.lstm.Wo,
        model.weekly_model.lstm.bf, model.weekly_model.lstm.bi, model.weekly_model.lstm.bc, model.weekly_model.lstm.bo,
        model.weekly_model.dense.W, model.weekly_model.dense.b,
    ])
    # Final Dense
    weights.extend([model.final_dense.W, model.final_dense.b])
    return weights

def set_model_weights(model, weights):
    idx = 0
    # Recent
    model.recent_model.lstm.Wf = weights[idx]; idx += 1
    model.recent_model.lstm.Wi = weights[idx]; idx += 1
    model.recent_model.lstm.Wc = weights[idx]; idx += 1
    model.recent_model.lstm.Wo = weights[idx]; idx += 1
    model.recent_model.lstm.bf = weights[idx]; idx += 1
    model.recent_model.lstm.bi = weights[idx]; idx += 1
    model.recent_model.lstm.bc = weights[idx]; idx += 1
    model.recent_model.lstm.bo = weights[idx]; idx += 1
    model.recent_model.dense.W = weights[idx]; idx += 1
    model.recent_model.dense.b = weights[idx]; idx += 1
    # Daily
    model.daily_model.lstm.Wf = weights[idx]; idx += 1
    model.daily_model.lstm.Wi = weights[idx]; idx += 1
    model.daily_model.lstm.Wc = weights[idx]; idx += 1
    model.daily_model.lstm.Wo = weights[idx]; idx += 1
    model.daily_model.lstm.bf = weights[idx]; idx += 1
    model.daily_model.lstm.bi = weights[idx]; idx += 1
    model.daily_model.lstm.bc = weights[idx]; idx += 1
    model.daily_model.lstm.bo = weights[idx]; idx += 1
    model.daily_model.dense.W = weights[idx]; idx += 1
    model.daily_model.dense.b = weights[idx]; idx += 1
    # Weekly
    model.weekly_model.lstm.Wf = weights[idx]; idx += 1
    model.weekly_model.lstm.Wi = weights[idx]; idx += 1
    model.weekly_model.lstm.Wc = weights[idx]; idx += 1
    model.weekly_model.lstm.Wo = weights[idx]; idx += 1
    model.weekly_model.lstm.bf = weights[idx]; idx += 1
    model.weekly_model.lstm.bi = weights[idx]; idx += 1
    model.weekly_model.lstm.bc = weights[idx]; idx += 1
    model.weekly_model.lstm.bo = weights[idx]; idx += 1
    model.weekly_model.dense.W = weights[idx]; idx += 1
    model.weekly_model.dense.b = weights[idx]; idx += 1
    # Final Dense
    model.final_dense.W = weights[idx]; idx += 1
    model.final_dense.b = weights[idx]; idx += 1

# --------- FL CLIENT ---------
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model_weights(model)

    def set_parameters(self, parameters):
        set_model_weights(model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for epoch in range(LOCAL_EPOCHS):
            total_loss = 0
            # --- TRAIN ---
            for i in range(len(X_recent_train)):
                targets = {
                    "recent": Y_train_scaled[i].reshape(-1, 1),
                    "daily": Y_train_scaled[i].reshape(-1, 1),
                    "weekly": Y_train_scaled[i].reshape(-1, 1),
                    "final": Y_train_scaled[i].reshape(-1, 1)
                }
                loss, _, _ = model.train_step(
                    X_recent_train[i],
                    X_daily_train[i],
                    X_weekly_train[i],
                    targets
                )
                total_loss += loss
            avg_loss = total_loss / len(X_recent_train)
            # --- VAL ---
            total_val_loss = 0
            for i in range(len(X_recent_val)):
                targets_val = {
                    "recent": Y_val_scaled[i].reshape(-1, 1),
                    "daily": Y_val_scaled[i].reshape(-1, 1),
                    "weekly": Y_val_scaled[i].reshape(-1, 1),
                    "final": Y_val_scaled[i].reshape(-1, 1)
                }
                outputs_val = model.forward(
                    X_recent_val[i],
                    X_daily_val[i],
                    X_weekly_val[i]
                )
                val_loss, _, _ = model.compute_loss(outputs_val, targets_val)
                total_val_loss += val_loss
            avg_val_loss = total_val_loss / len(X_recent_val)
            print(f"[Client {client_id}] Epoch {epoch+1}/{LOCAL_EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        return self.get_parameters(config), len(X_recent_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        total_loss = 0
        predictions = []
        true_values = []
        for i in range(len(X_test_recent)):
            outputs = model.forward(
                X_test_recent[i],
                X_test_daily[i],
                X_test_weekly[i]
            )
            targets = {
                "recent": Y_test_scaled[i].reshape(-1, 1),
                "daily": Y_test_scaled[i].reshape(-1, 1),
                "weekly": Y_test_scaled[i].reshape(-1, 1),
                "final": Y_test_scaled[i].reshape(-1, 1)
            }
            loss, _, _ = model.compute_loss(outputs, targets)
            total_loss += loss
            predictions.append(outputs["final_output"].flatten())
            true_values.append(Y_test_scaled[i])
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        pred_inv = scaler_y.inverse_transform(predictions)
        true_inv = scaler_y.inverse_transform(true_values)
        rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
        r2 = r2_score(true_inv, pred_inv)
        mae = np.mean(np.abs(true_inv - pred_inv))
        mape = np.mean(np.abs((true_inv - pred_inv) / (true_inv + 1e-8))) * 100
        avg_loss = total_loss / len(X_test_recent)
        print(f"[Client {client_id}] Test Loss: {avg_loss:.4f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        return avg_loss, len(X_test_recent), {
            "rmse": float(rmse),
            "r2": float(r2),
            "mae": float(mae),
            "mape": float(mape),
            "client_id": client_id
        }

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FLClient())
