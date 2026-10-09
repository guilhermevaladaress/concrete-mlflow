import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    # Não semear o `random` global: o OpenTelemetry (tracing do MLflow) gera
    # os trace IDs com ele, e todas as runs receberiam o mesmo ID (colisão).
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_concrete():
    ds = fetch_ucirepo(id=165)
    X = ds.data.features.to_numpy(dtype=np.float32)
    y = ds.data.targets.to_numpy(dtype=np.float32).reshape(-1)
    return X, y


def split_data(X, y, seed):
    """60/20/20. O scaler é ajustado depois, só no treino."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(in_features, hidden_sizes):
    return MLP(in_features, hidden_sizes)


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
