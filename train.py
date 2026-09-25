"""Entrena un MLP pequeño sobre data/digits.csv y registra el experimento en MLflow.

El servidor MLflow se elige con MLFLOW_TRACKING_URI (local o DagsHub).

Uso:
    python train.py --hidden 64 --lr 1e-3 --epochs 40
    python train.py --hidden 128 64 --dropout 0.2
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLP

DATA = Path("data/digits.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[64])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", default="mlp-digits")
    return p.parse_args()


def data_version():
    """md5 del dataset según DVC, para enlazar cada run con su versión de datos."""
    dvc_file = DATA.with_suffix(".csv.dvc")
    if not dvc_file.exists():
        return "untracked"
    return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]


def to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        loss = loss_fn(logits, torch.tensor(y)).item()
    return loss, logits.argmax(1).numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    df = pd.read_csv(DATA)
    X = df.drop(columns="label").values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

    model = MLP(X.shape[1], args.hidden, n_classes=10, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = to_loader(X_train, y_train, args.batch_size, shuffle=True)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "hidden": "-".join(map(str, args.hidden)),
            "n_params": sum(p.numel() for p in model.parameters()),
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "seed": args.seed,
        })
        mlflow.set_tag("data_md5", data_version())
        mlflow.set_tag("n_samples", len(df))

        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            val_loss, val_pred = evaluate(model, X_val, y_val, loss_fn)
            mlflow.log_metrics({
                "train_loss": total / len(X_train),
                "val_loss": val_loss,
                "val_accuracy": accuracy_score(y_val, val_pred),
            }, step=epoch)

        _, test_pred = evaluate(model, X_test, y_test, loss_fn)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        mlflow.log_metrics({"test_accuracy": test_acc, "test_f1_macro": test_f1})

        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, test_pred, ax=ax, colorbar=False)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.pytorch.log_model(
            model,
            name="mlp-model",
            input_example=X_test[:5].astype("float32"),
        )

        print(f"hidden={args.hidden} lr={args.lr} dropout={args.dropout} "
              f"-> test_acc={test_acc:.4f} f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
