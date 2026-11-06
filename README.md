# Go Neural Network Projects

This repository contains two CLI-first machine learning workflows in Go:

- an Iris multiclass classifier
- a credit-card fraud classifier built on the shared `internal/irisnn` neural-network core

## Layout

- `iris-datasets/`
  - `Iris.csv`
  - `train.csv`
  - `validation.csv`
  - `test.csv`
  - `model.json`
  - `predictions.csv`
- `credit-card-fraud-datasets/`
  - `transactions_train.csv`
  - `transactions_train_split.csv`
  - `transactions_validation.csv`
  - `transactions_test.csv`
  - `transactions_model.json`
  - `transactions_predictions.csv`
- `internal/irisnn/`
  - shared neural-network core, scaler, binary helpers, and model I/O
- `internal/iris/`
  - Iris-specific dataset loading and evaluation helpers
- `tests/`
  - integration and package tests

## Build

This repo has multiple standalone `main` programs in the root, so build each CLI explicitly:

```bash
env GOCACHE=/tmp/go-cache go build -o /tmp/iris-split train_test_split.go
env GOCACHE=/tmp/go-cache go build -o /tmp/iris-train neural_network.go
env GOCACHE=/tmp/go-cache go build -o /tmp/iris-predict predict.go

env GOCACHE=/tmp/go-cache go build -o /tmp/txsplit transactions_split.go
env GOCACHE=/tmp/go-cache go build -o /tmp/txfraud transactions_fraud.go
```

You can also use `go run ...` directly in all examples below.

## Iris Workflow

Split the dataset:

```bash
go run train_test_split.go \
  --input iris-datasets/Iris.csv \
  --train iris-datasets/train.csv \
  --validation iris-datasets/validation.csv \
  --test iris-datasets/test.csv \
  --train-ratio 0.7 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --workers 4
```

Train the model:

```bash
go run neural_network.go \
  --train-input iris-datasets/train.csv \
  --validation-input iris-datasets/validation.csv \
  --test-input iris-datasets/test.csv \
  --model-out iris-datasets/model.json \
  --epochs 100 \
  --learning-rate 0.15 \
  --hidden-neurons 10 \
  --seed 42
```

Predict one sample:

```bash
go run predict.go \
  --model iris-datasets/model.json \
  --sepal-length 5.1 \
  --sepal-width 3.5 \
  --petal-length 1.4 \
  --petal-width 0.2
```

Run batch prediction:

```bash
go run predict.go \
  --model iris-datasets/model.json \
  --batch-input iris-datasets/test.csv \
  --batch-output iris-datasets/predictions.csv \
  --workers 4 \
  --benchmark-repeats 3
```

## Credit-Card Fraud Workflow

Create train, validation, and test splits:

```bash
go run transactions_split.go \
  --input credit-card-fraud-datasets/transactions_train.csv \
  --train-output credit-card-fraud-datasets/transactions_train_split.csv \
  --validation-output credit-card-fraud-datasets/transactions_validation.csv \
  --test-output credit-card-fraud-datasets/transactions_test.csv \
  --validation-ratio 0.2 \
  --test-ratio 0.1 \
  --workers 8
```

Train the fraud neural network:

```bash
go run transactions_fraud.go \
  --mode train \
  --train-input credit-card-fraud-datasets/transactions_train_split.csv \
  --validation-input credit-card-fraud-datasets/transactions_validation.csv \
  --test-input credit-card-fraud-datasets/transactions_test.csv \
  --model credit-card-fraud-datasets/transactions_model.json \
  --epochs 10 \
  --learning-rate 0.01 \
  --hidden-neurons 16 \
  --seed 42 \
  --threshold-step 0.01
```

Predict one transaction:

```bash
go run transactions_fraud.go \
  --mode predict-one \
  --model credit-card-fraud-datasets/transactions_model.json \
  --step 1 \
  --type TRANSFER \
  --amount 5000 \
  --oldbalance-orig 5000 \
  --newbalance-orig 0 \
  --name-dest C999 \
  --oldbalance-dest 0 \
  --newbalance-dest 0
```

Predict a CSV with concurrency:

```bash
go run transactions_fraud.go \
  --mode predict-csv \
  --model credit-card-fraud-datasets/transactions_model.json \
  --input credit-card-fraud-datasets/transactions_train.csv \
  --output credit-card-fraud-datasets/transactions_predictions.csv \
  --max-rows 250000 \
  --workers 8
```

Predict the same CSV without concurrency:

```bash
go run transactions_fraud.go \
  --mode predict-csv \
  --model credit-card-fraud-datasets/transactions_model.json \
  --input credit-card-fraud-datasets/transactions_train.csv \
  --output credit-card-fraud-datasets/transactions_predictions.csv \
  --max-rows 250000 \
  --workers 1
```

## Metrics

The fraud training command prints metrics for train, validation, and test splits. These are the main ones to pay attention to:

- `loss`: binary cross-entropy over the fraud-class probability
- `accuracy`: fraction of rows classified correctly at the chosen threshold
- `precision`: of the rows predicted as fraud, how many were actually fraud
- `recall`: of the actual fraud rows, how many the model found
- `F1`: balance between precision and recall
- `F2`: like F1, but weights recall more heavily; this is the metric used to choose the fraud threshold
- `threshold`: the fraud probability cutoff used to convert class-1 probability into label `0` or `1`

Example training output:

```text
Training complete - train loss 0.622566 acc 0.9229 precision 0.0385 recall 0.3333 f2 0.1316 | val loss 0.633775 acc 0.8850 precision 0.0435 recall 0.5000 f2 0.1613 | test loss 0.629822 acc 0.8700 precision 0.0000 recall 0.0000 f2 0.0000 | threshold 0.53
```

How to read that:

- `accuracy` can look high on imbalanced fraud data even when fraud detection is weak
- `precision`, `recall`, and especially `F2` matter more than raw accuracy for the fraud task
- if `test recall` is low, the model is missing fraud cases even if overall accuracy is still high

The Iris workflow reports:

- `loss`
- multiclass `accuracy`

because Iris is a balanced multiclass problem and does not use thresholded fraud metrics.

## Concurrency Notes

Concurrency is used in these places:

- `train_test_split.go`: worker goroutines for Iris row parsing and encoding
- `predict.go`: worker goroutines for Iris batch inference
- `transactions_split.go`: worker goroutines for transaction row validation before deterministic split assignment
- `transactions_fraud.go`: worker goroutines for batch fraud prediction

On the current local machine used during development:

- fraud split benchmark on `6351193` rows:
  - `workers=1`: about `15.05s`
  - `workers=8`: about `12.24s`
- fraud batch prediction benchmark on `6351193` rows:
  - `workers=1`: about `30.26s`
  - `workers=8`: about `15.86s`

That means concurrency helps much more in fraud batch prediction than in the current split implementation.

## Testing

Run the full test suite:

```bash
env GOCACHE=/tmp/go-cache go test ./tests
```

Run only the fraud workflow test:

```bash
env GOCACHE=/tmp/go-cache go test ./tests -run TestTransactionsFraudTrainAndPredictCSV
```

## Fraud training loss

Both `transactions_fraud.go` and `transactions_fraud_nn.go` train with binary
cross-entropy (BCE) and one sigmoid output representing P(fraud). Labels remain
0 for non-fraud and 1 for fraud. Console loss is labelled `bce`; saved JSON uses
`loss`. The shared network records `config.loss = "bce"`; omitted loss settings
retain the Iris MSE behavior.

The shared trainer logs BCE before each epoch's update; its final summary uses
the updated model. The standalone trainer evaluates after each update. BCE does
not depend on the decision threshold; validation F2 still selects that threshold.
Both programs can predict with their existing two-output model formats using
output column 1. New models use column 0. Retraining is required to adopt BCE;
changing the loss does not guarantee higher fraud recall.

Standalone trainer checks:

```bash
env GOCACHE=/tmp/go-cache go test transactions_fraud_nn.go transactions_fraud_nn_test.go
```

### Average Precision (AP)

Both fraud programs report `ap` for each evaluated split and save it as
`average_precision` in the split summary. This is the non-interpolated
precision–recall summary: `AP = sum((recall increase) * precision)`, not
trapezoidal PR-AUC. Higher values indicate better ranking of fraud cases.
Equal scores are evaluated together; all tied scores yield the positive-class
fraction. No positives yields 0; all positives yields 1.

AP uses fraud probabilities across all score thresholds and does not depend on
the chosen decision threshold. BCE training and validation F2 threshold selection
are unchanged. Calculation sorts a copy of the scores, costing O(n log n) time
and O(n) memory per evaluated split (each epoch in the standalone trainer).
Old model files remain readable but do not contain historical AP measurements;
missing values load as zero and must not be interpreted as measured AP.

### Dataset class balance

Run `go run ./helpers` from the repository root to print total rows, fraud and
non-fraud counts, and their percentages for the training, validation, and test
splits. It streams the CSV files without modifying them. Custom paths can be
supplied with `--train-input`, `--validation-input`, and `--test-input`.

The helper function `CalculateFraudStats` is in `helpers/dataset_balance.go`.
It requires an `isFraud` column containing only `0` or `1` and rejects empty
datasets instead of reporting misleading percentages.
