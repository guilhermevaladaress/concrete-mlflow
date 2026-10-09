import argparse
import os
import platform
import subprocess
import tempfile

import mlflow
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import StandardScaler

from common import (
    build_model,
    get_device,
    load_concrete,
    regression_metrics,
    set_seed,
    split_data,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def version_tags():
    """Versão do código (commit) e do ambiente, para rastrear cada run."""
    status = git("status", "--porcelain")
    tags = {
        "git_commit": git("rev-parse", "HEAD") or "desconhecido",
        "git_dirty": "desconhecido" if status is None else str(bool(status)).lower(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mlflow_version": mlflow.__version__,
        "sklearn_version": sklearn.__version__,
    }
    if status:
        tags["git_dirty_files"] = status[:5000]
    return tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 1. CONFIGURAR
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Usando o dispositivo: {device}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("concrete-mlp")

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params({
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "epochs": cfg["epochs"],
            "hidden_sizes": str(cfg["hidden_sizes"]),
            "seed": cfg["seed"],
            "optimizer": "Adam",
            "loss": "MSELoss",
            "split": "60/20/20",
            "device": str(device),
        })
        mlflow.set_tags(version_tags())
        mlflow.log_artifact(os.path.join(ROOT, "requirements.txt"))
        mlflow.log_dict(cfg, "config.json")

        with mlflow.start_span(name="pipeline"):
            # 2. PREPARAR
            with mlflow.start_span(name="preparar"):
                X, y = load_concrete()
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    X, y, cfg["seed"]
                )
                x_scaler = StandardScaler().fit(X_train)                  # só no treino
                y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))   # só no treino

                def to_t(a):
                    return torch.tensor(a, dtype=torch.float32, device=device)

                X_train_t = to_t(x_scaler.transform(X_train))
                y_train_t = to_t(y_scaler.transform(y_train.reshape(-1, 1)))
                X_val_t = to_t(x_scaler.transform(X_val))
                y_val_t = to_t(y_scaler.transform(y_val.reshape(-1, 1)))

                preprocessing = {
                    "x_mean": x_scaler.mean_.tolist(),
                    "x_scale": x_scaler.scale_.tolist(),
                    "y_mean": float(y_scaler.mean_[0]),
                    "y_scale": float(y_scaler.scale_[0]),
                    "n_train": int(len(y_train)),
                    "n_val": int(len(y_val)),
                    "n_test": int(len(y_test)),
                    "split": "60/20/20",
                    "seed": cfg["seed"],
                    "scaler_ajustado_em": "treino",
                }

            # 3. TREINAR
            with mlflow.start_span(name="treinar"):
                model = build_model(X_train.shape[1], cfg["hidden_sizes"]).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(
                    model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
                )

                for epoch in range(1, cfg["epochs"] + 1):
                    model.train()
                    outputs = model(X_train_t)
                    loss = criterion(outputs, y_train_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_out = model(X_val_t)
                        val_loss = criterion(val_out, y_val_t)

                    val_pred = y_scaler.inverse_transform(val_out.cpu().numpy()).ravel()
                    m = regression_metrics(y_val, val_pred)  # em MPa

                    mlflow.log_metrics(
                        {
                            "train_loss": loss.item(),
                            "val_loss": val_loss.item(),
                            "val_rmse": m["rmse"],
                            "val_mae": m["mae"],
                            "val_r2": m["r2"],
                        },
                        step=epoch,
                    )

                    if epoch % 100 == 0:
                        print(
                            f"Época [{epoch}/{cfg['epochs']}] "
                            f"train_loss={loss.item():.4f} val_loss={val_loss.item():.4f} "
                            f"val_rmse={m['rmse']:.2f} MPa val_r2={m['r2']:.3f}"
                        )

            # 4. VALIDAR (métricas finais e artefatos; sem teste aqui)
            with mlflow.start_span(name="validar"):
                mlflow.log_metrics({
                    "final_val_rmse": m["rmse"],
                    "final_val_mae": m["mae"],
                    "final_val_r2": m["r2"],
                })
                mlflow.log_dict(preprocessing, "preprocessing.json")
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "model.pt")
                    torch.save(model.state_dict(), path)
                    mlflow.log_artifact(path, artifact_path="model")

        print(f"Run concluída. run_id = {run.info.run_id}")


if __name__ == "__main__":
    main()
