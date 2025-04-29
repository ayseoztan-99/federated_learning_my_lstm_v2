#model_numpy_lstm.py

import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Ağırlıklar ve biaslar
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))*1.0
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        # Gradyanlar
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, out):
        return out * (1 - out)
    def tanh(self, x):
        return np.tanh(x)
    def tanh_derivative(self, out):
        return 1 - out**2

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack((h_prev, x))
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_hat = self.tanh(np.dot(self.Wc, combined) + self.bc)
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        c = f * c_prev + i * c_hat
        h = o * self.tanh(c)
        cache = (x, h_prev, c_prev, f, i, c_hat, o, c, combined)
        return h, c, cache

    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, f, i, c_hat, o, c, combined = cache
        tanh_c = self.tanh(c)
        do = dh * tanh_c
        do_pre = do * self.sigmoid_derivative(o)
        self.dWo += np.dot(do_pre, combined.T)
        self.dbo += do_pre

        dc_total = dc + dh * o * self.tanh_derivative(tanh_c)
        df = dc_total * c_prev
        df_pre = df * self.sigmoid_derivative(f)
        self.dWf += np.dot(df_pre, combined.T)
        self.dbf += df_pre

        di = dc_total * c_hat
        di_pre = di * self.sigmoid_derivative(i)
        self.dWi += np.dot(di_pre, combined.T)
        self.dbi += di_pre

        dc_hat = dc_total * i
        dc_hat_pre = dc_hat * self.tanh_derivative(c_hat)
        self.dWc += np.dot(dc_hat_pre, combined.T)
        self.dbc += dc_hat_pre

        dc_prev = dc_total * f
        dcombined = (np.dot(self.Wf.T, df_pre) +
                     np.dot(self.Wi.T, di_pre) +
                     np.dot(self.Wc.T, dc_hat_pre) +
                     np.dot(self.Wo.T, do_pre))
        dh_prev = dcombined[:self.hidden_size]
        dx = dcombined[self.hidden_size:]
        return dh_prev, dc_prev, dx

    def update_weights(self, learning_rate):
        self.Wf -= learning_rate * self.dWf
        self.Wi -= learning_rate * self.dWi
        self.Wc -= learning_rate * self.dWc
        self.Wo -= learning_rate * self.dWo
        self.bf -= learning_rate * self.dbf
        self.bi -= learning_rate * self.dbi
        self.bc -= learning_rate * self.dbc
        self.bo -= learning_rate * self.dbo
        # Gradyanları sıfırla
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)

class Dense:
    # NumPy ile Dense (fully connected) katmanı
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * 0.1
        self.b = np.zeros((output_size, 1))
    def forward(self, x):
        return np.dot(self.W, x) + self.b
    def backward(self, dout, x, lr):
        self.dW = np.dot(dout, x.T)
        self.db = dout
        self.W -= lr * self.dW
        self.b -= lr * self.db

class FinalDense:
    # Son katman (Dense)
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * 0.1
        self.b = np.zeros((output_size, 1))
    def forward(self, x):
        return np.dot(self.W, x) + self.b
    def backward(self, dout, x, lr):
        self.dW = np.dot(dout, x.T)
        self.db = dout
        self.W -= lr * self.dW
        self.b -= lr * self.db

class CustomLSTMModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTM(input_size, hidden_size)
        self.dense = Dense(hidden_size, output_size)
    def forward(self, X):
        h = np.zeros((self.lstm.hidden_size, 1))
        c = np.zeros((self.lstm.hidden_size, 1))
        self.caches = []
        for t in range(len(X)):
            x = X[t].reshape(-1, 1)
            h, c, cache = self.lstm.forward(x, h, c)
            self.caches.append(cache)
        self.h = h
        y = self.dense.forward(h)
        return y
    def backward(self, dy, learning_rate):
        self.dense.backward(dy, self.h, learning_rate)
        dh = np.dot(self.dense.W.T, dy)
        dc = np.zeros_like(self.caches[0][2])
        for t in reversed(range(len(self.caches))):
            dh, dc, dx = self.lstm.backward(dh, dc, self.caches[t])
        self.lstm.update_weights(learning_rate)
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    def mse_loss_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size
    def mae_metric(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

class MultiLSTMModel:
    def __init__(self, recent_model, daily_model, weekly_model, tp):
        self.recent_model = recent_model
        self.daily_model = daily_model
        self.weekly_model = weekly_model
        self.final_dense = FinalDense(tp * 3, tp)  # Çıkışları birleştirip final dense ile sonuca gider
        self.learning_rate = 0.001

    def forward(self, recent_data, daily_data, weekly_data):
        out_recent = self.recent_model.forward(recent_data)
        out_daily = self.daily_model.forward(daily_data)
        out_weekly = self.weekly_model.forward(weekly_data)
        # Çıktıları birleştir
        merged = np.vstack([out_recent, out_daily, out_weekly])
        final_output = self.final_dense.forward(merged)
        return {
            "output_recent": out_recent,
            "output_daily": out_daily,
            "output_weekly": out_weekly,
            "merged": merged,
            "final_output": final_output
        }

    def compute_loss(self, outputs, targets):
        losses = {
            "output_recent": self.recent_model.mse_loss(outputs["output_recent"], targets["recent"]),
            "output_daily": self.daily_model.mse_loss(outputs["output_daily"], targets["daily"]),
            "output_weekly": self.weekly_model.mse_loss(outputs["output_weekly"], targets["weekly"]),
            "final_output": self.recent_model.mse_loss(outputs["final_output"], targets["final"])
        }
        maes = {
            "output_recent": self.recent_model.mae_metric(outputs["output_recent"], targets["recent"]),
            "output_daily": self.daily_model.mae_metric(outputs["output_daily"], targets["daily"]),
            "output_weekly": self.weekly_model.mae_metric(outputs["output_weekly"], targets["weekly"]),
            "final_output": self.recent_model.mae_metric(outputs["final_output"], targets["final"])
        }
        loss_weights = {
            "output_recent": 0.1,
            "output_daily": 0.1,
            "output_weekly": 0.1,
            "final_output": 0.7
        }
        total_loss = sum(losses[key] * loss_weights[key] for key in losses)
        return total_loss, losses, maes

    def train_step(self, recent_data, daily_data, weekly_data, targets):
        outputs = self.forward(recent_data, daily_data, weekly_data)
        total_loss, losses, maes = self.compute_loss(outputs, targets)
        dy_recent = self.recent_model.mse_loss_derivative(outputs["output_recent"], targets["recent"])
        dy_daily = self.daily_model.mse_loss_derivative(outputs["output_daily"], targets["daily"])
        dy_weekly = self.weekly_model.mse_loss_derivative(outputs["output_weekly"], targets["weekly"])
        dy_final = self.recent_model.mse_loss_derivative(outputs["final_output"], targets["final"])

        # Final Dense backward
        self.final_dense.backward(dy_final, outputs["merged"], self.learning_rate)
        dmerged = np.dot(self.final_dense.W.T, dy_final)
        # Her bir modele split et
        tp = outputs["output_recent"].shape[0]
        dmerged_recent = dmerged[0:tp]
        dmerged_daily = dmerged[tp:2*tp]
        dmerged_weekly = dmerged[2*tp:]
        self.recent_model.backward(dy_recent * 0.1 + dmerged_recent * 0.7, self.learning_rate)
        self.daily_model.backward(dy_daily * 0.1 + dmerged_daily * 0.7, self.learning_rate)
        self.weekly_model.backward(dy_weekly * 0.1 + dmerged_weekly * 0.7, self.learning_rate)
        return total_loss, losses, maes

def build_multi_lstm_model(th, td, tw, tp):
    recent_model = CustomLSTMModel(1, 64, tp)
    daily_model = CustomLSTMModel(1, 32, tp)
    weekly_model = CustomLSTMModel(1, 32, tp)
    return MultiLSTMModel(recent_model, daily_model, weekly_model, tp)
