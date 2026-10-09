import argparse

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from common import (
    build_model,
    get_device,
    load_concrete,
    regression_metrics,
    split_data,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    run = client.get_run(args.run_id)
    if "test_rmse" in run.data.metrics:
        raise SystemExit(
            "Esta run já tem métricas de teste. O teste só é avaliado uma vez."
        )

    cfg = mlflow.artifacts.load_dict(f"runs:/{args.run_id}/config.json")
    prep = mlflow.artifacts.load_dict(f"runs:/{args.run_id}/preprocessing.json")
    model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{args.run_id}/model/model.pt"
    )

    device = get_device()
    X, y = load_concrete()
    _, _, X_test, _, _, y_test = split_data(X, y, cfg["seed"])
    assert len(y_test) == prep["n_test"], "O split não bate com o da run."

    x_mean = np.array(prep["x_mean"], dtype=np.float32)
    x_scale = np.array(prep["x_scale"], dtype=np.float32)
    X_test_s = (X_test - x_mean) / x_scale

    model = build_model(X_test.shape[1], cfg["hidden_sizes"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(torch.tensor(X_test_s, dtype=torch.float32, device=device))
    pred = out.cpu().numpy().ravel() * prep["y_scale"] + prep["y_mean"]

    m = regression_metrics(y_test, pred)
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics(
            {"test_rmse": m["rmse"], "test_mae": m["mae"], "test_r2": m["r2"]}
        )

    print(
        f"Teste: RMSE={m['rmse']:.2f} MPa | MAE={m['mae']:.2f} MPa | R2={m['r2']:.3f}"
    )


if __name__ == "__main__":
    main()
