"""Inspectable PyTorch replica of the balance-free Go fraud pipeline."""

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
FEATURES = (
    ["log_amount_z", "transfer_amount", "cashout_amount", "hour_sin", "hour_cos"]
    + ["type_" + t for t in TYPES]
    + ["dest_is_merchant"]
)
COLUMNS = ["step", "type", "amount", "nameDest", "isFraud"]


@dataclass
class Config:
    train_input: str = str(
        ROOT / "credit-card-fraud-datasets/transactions_train_split.csv"
    )
    validation_input: str = str(
        ROOT / "credit-card-fraud-datasets/transactions_validation.csv"
    )
    test_input: str = str(ROOT / "credit-card-fraud-datasets/transactions_test.csv")
    train_size: int = 200000
    validation_size: int = 50000
    test_size: int = 50000
    epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 0.001
    hidden: int = 64
    hidden2: int = 32
    optimizer: str = "adam"
    positive_weight: str = "auto"
    seed: int = 42
    device: str = "cpu"
    output: str = str(ROOT / "experiment/runs")


def preprocess(frame):
    frame = frame.copy()
    for col in ["step", "amount", "isFraud"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col]).all():
            raise ValueError(f"{col} must be finite")
    if not frame.isFraud.isin([0, 1]).all():
        raise ValueError("isFraud must be binary")
    if (
        (frame.amount < 0).any()
        or (frame.step < 0).any()
        or (frame.step % 1 != 0).any()
    ):
        raise ValueError(
            "amount must be nonnegative; step must be a nonnegative integer"
        )
    frame["type"] = frame["type"].str.upper()
    if not frame["type"].isin(TYPES).all():
        raise ValueError("unsupported transaction type")
    return frame


def load_sample(path, size, seed=42):
    if size < 0:
        raise ValueError("sample size must be nonnegative")
    labels = pd.read_csv(path, usecols=["isFraud"]).isFraud.to_numpy()
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("empty dataset or invalid labels")
    indices = np.arange(len(labels))
    if 0 < size < len(labels):
        indices, _ = train_test_split(
            indices, train_size=size, stratify=labels, random_state=seed
        )
    selected = np.sort(indices)
    parts = []
    for chunk in pd.read_csv(
        path, usecols=COLUMNS, chunksize=100000, keep_default_na=False
    ):
        parts.append(chunk.loc[chunk.index.isin(selected)])
    frame = preprocess(pd.concat(parts))
    print(f"{Path(path).name}: {len(frame)} rows, {int(frame.isFraud.sum())} fraud")
    return frame


def fit_scaler(train):
    return StandardScaler().fit(np.log1p(train[["amount"]]))


def build_features(frame, scaler):
    frame = preprocess(frame)
    z = scaler.transform(np.log1p(frame[["amount"]])).ravel()
    angle = 2 * np.pi * (frame.step.to_numpy() % 24) / 24
    values = [
        z,
        z * (frame["type"] == "TRANSFER"),
        z * (frame["type"] == "CASH_OUT"),
        np.sin(angle),
        np.cos(angle),
    ]
    values += [(frame["type"] == t).to_numpy() for t in TYPES]
    values += [frame.nameDest.str.startswith("M").to_numpy()]
    return pd.DataFrame(np.column_stack(values), columns=FEATURES, index=frame.index)


def build_model(config):
    torch.manual_seed(config.seed)
    model = nn.Sequential(
        nn.Linear(11, config.hidden),
        nn.ReLU(),
        nn.Linear(config.hidden, config.hidden2),
        nn.ReLU(),
        nn.Linear(config.hidden2, 1),
    )
    for layer in [model[0], model[2]]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.uniform_(model[4].weight, -0.5, 0.5)
    for layer in [model[0], model[2], model[4]]:
        nn.init.zeros_(layer.bias)
    return model.to(config.device)


def build_optimizer(model, config):
    if config.optimizer not in ["adam", "sgd"]:
        raise ValueError("optimizer must be adam or sgd")
    cls = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    return cls(model.parameters(), lr=config.learning_rate)


def make_loader(features, labels, batch_size=256, seed=42):
    dataset = TensorDataset(
        torch.tensor(np.asarray(features), dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.float32).reshape(-1, 1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(model, loader, optimizer, positive_weight):
    model.train()
    device = next(model.parameters()).device
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=device)
    )
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, features, batch_size=8192):
    model.eval()
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    device = next(model.parameters()).device
    return (
        torch.cat(
            [model(b.to(device)).sigmoid().cpu().flatten() for b in x.split(batch_size)]
        )
        .numpy()
        .astype(float)
    )


def evaluate(labels, probabilities, threshold=0.3, positive_weight=1.0, stats=False):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (
        not len(y)
        or y.shape != p.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("invalid labels/probabilities")
    if not 0 <= threshold <= 1:
        raise ValueError("invalid threshold")
    tn, fp, fn, tp = map(
        int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    )
    q = np.clip(p, 1e-9, 1 - 1e-9)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result = dict(
        loss=float(np.mean(-y * np.log(q) - (1 - y) * np.log1p(-q))),
        weighted_bce=float(
            np.mean(-positive_weight * y * np.log(q) - (1 - y) * np.log1p(-q))
        ),
        average_precision=float(average_precision_score(y, p)) if y.sum() else 0.0,
        accuracy=(tp + tn) / len(y),
        precision=precision,
        recall=recall,
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        f2=5 * tp / max(5 * tp + fp + 4 * fn, 1),
        true_pos=tp,
        false_pos=fp,
        true_neg=tn,
        false_neg=fn,
        rows=len(y),
        positive_rate=float(y.mean()),
        threshold=float(threshold),
    )
    if stats:
        result["probability_stats"] = {}
        for label, name in [(0, "legitimate"), (1, "fraud")]:
            a = p[y == label]
            result["probability_stats"][name] = (
                dict(
                    zip(
                        ["mean", "median", "p90", "p99"],
                        map(float, [a.mean(), *np.quantile(a, [0.5, 0.9, 0.99])]),
                    )
                )
                if len(a)
                else None
            )
    return result


def select_threshold(labels, probabilities):
    evaluate(labels, probabilities)
    if not np.sum(labels):
        raise ValueError("validation requires fraud cases; increase sample size")
    precision, recall, thresholds = precision_recall_curve(
        labels, probabilities, drop_intermediate=False
    )
    best = (-1.0, -1.0, -1.0, 0.0)
    for p, r, t in zip(precision[:-1][::-1], recall[:-1][::-1], thresholds[::-1]):
        candidate = (5 * p * r / (4 * p + r) if 4 * p + r else 0.0, r, p, -t)
        for a, b in zip(candidate[:3], best[:3]):
            if a > b + 1e-12:
                best = candidate
                break
            if abs(a - b) > 1e-12:
                break
        else:
            if candidate[3] > best[3]:
                best = candidate
    return float(-best[3])


def save_checkpoint(path, model, optimizer, scaler, threshold, config):
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            features=FEATURES,
            threshold=threshold,
            config=asdict(config),
        ),
        path,
    )


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["features"] != FEATURES:
        raise ValueError("incompatible features")
    config = Config(**saved["config"])
    config.device = "cpu"
    model = build_model(config)
    model.load_state_dict(saved["model"])
    optimizer = build_optimizer(model, config)
    optimizer.load_state_dict(saved["optimizer"])
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.array(saved["mean"]), np.array(saved["scale"])
    scaler.n_features_in_ = 1
    scaler.feature_names_in_ = np.array(["amount"], dtype=object)
    return model, optimizer, scaler, saved["threshold"], config


def run(config):
    if (
        min(config.epochs, config.batch_size, config.hidden, config.hidden2) <= 0
        or not np.isfinite(config.learning_rate)
        or config.learning_rate <= 0
    ):
        raise ValueError(
            "epochs, batch size, widths and learning rate must be positive"
        )
    start = time.perf_counter()
    frames = {
        name: load_sample(
            getattr(config, name + "_input"),
            getattr(config, name + "_size"),
            config.seed,
        )
        for name in ["train", "validation", "test"]
    }
    y = {name: f.isFraud.to_numpy() for name, f in frames.items()}
    if len(np.unique(y["train"])) != 2 or not y["validation"].sum():
        raise ValueError(
            "increase sample size: training needs both classes and validation needs fraud"
        )
    weight = (
        float((y["train"] == 0).sum() / y["train"].sum())
        if config.positive_weight == "auto"
        else float(config.positive_weight)
    )
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("positive weight must be finite and positive")
    scaler = fit_scaler(frames["train"])
    x = {name: build_features(f, scaler) for name, f in frames.items()}
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    loader = make_loader(x["train"], y["train"], config.batch_size, config.seed)
    history = []
    for epoch in range(1, config.epochs + 1):
        train_epoch(model, loader, optimizer, weight)
        for name in ["train", "validation"]:
            metrics = evaluate(y[name], predict(model, x[name]), positive_weight=weight)
            history.append(dict(epoch=epoch, split=name, **metrics))
            print(
                f'{epoch} {name}: BCE {metrics["loss"]:.5f}, AP {metrics["average_precision"]:.5f}'
            )
    probabilities = {name: predict(model, features) for name, features in x.items()}
    threshold = select_threshold(y["validation"], probabilities["validation"])
    final = {
        name: evaluate(y[name], p, threshold, weight, True)
        for name, p in probabilities.items()
    }
    table = [
        evaluate(y["validation"], probabilities["validation"], t, weight)
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
    ]
    directory = Path(config.output) / time.strftime("%Y%m%d-%H%M%S")
    directory = directory.with_name(
        directory.name + "-" + str(time.time_ns() % 1000000000)
    )
    directory.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    pd.DataFrame(table).to_csv(directory / "validation_thresholds.csv", index=False)
    np.savez_compressed(
        directory / "sample_indices.npz",
        **{k: f.index.to_numpy() for k, f in frames.items()},
    )
    metadata = dict(
        config=asdict(config),
        positive_weight=weight,
        elapsed_seconds=time.perf_counter() - start,
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    for filename, value in [("config.json", metadata), ("metrics.json", final)]:
        (directory / filename).write_text(json.dumps(value, indent=2, allow_nan=False))
    save_checkpoint(directory / "model.pt", model, optimizer, scaler, threshold, config)
    print(f"Results: {directory}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in asdict(Config()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), default=value, type=type(value)
        )
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
