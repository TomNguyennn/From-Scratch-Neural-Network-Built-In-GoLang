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
  --seed 42
```

Predict one transaction:

```bash
go run transactions_fraud.go \
  --mode predict-one \
  --model credit-card-fraud-datasets/transactions_model.json \
  --step 1 \
  --type TRANSFER \
  --amount 5000 \
  --name-dest C999
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
The historical standalone trainer retains two-output prediction support using
column 1. The updated trainer requires its new balance-free feature schema.
New single-output models use column 0. Retraining is required to adopt BCE;
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
Compatible older model files may lack historical AP measurements;
missing values load as zero and must not be interpreted as measured AP.

### Dataset class balance

Run `go run ./helpers` from the repository root to print total rows, fraud and
non-fraud counts, and their percentages for the training, validation, and test
splits. It streams the CSV files without modifying them. Custom paths can be
supplied with `--train-input`, `--validation-input`, and `--test-input`.

The helper function `CalculateFraudStats` is in `helpers/dataset_balance.go`.
It requires an `isFraud` column containing only `0` or `1` and rejects empty
datasets instead of reporting misleading percentages.

### Balance-free fraud features

`transactions_fraud.go` uses 11 features: `log_amount_z`, `transfer_amount`,
`cashout_amount`, `hour_sin`, `hour_cos`, five transaction-type indicators,
and `dest_is_merchant`. Log amount is `log1p(amount)`, standardized using training
statistics only (constant amounts use standard deviation 1). Each type-specific
amount equals `log_amount_z` for that type, otherwise zero. Cyclic features use
`step % 24` as simulation hour and are not standardized. Balance cells in CSVs
are ignored, including blanks. Balance command-line flags are removed.

Old feature schemas are rejected: retrain before using this program for prediction.
`transactions_fraud_nn.go` is preserved as the historical balance-based reference;
its existing feature-packing inconsistency remains and limits baseline comparisons.

Feature tests: `go test transactions_fraud.go transactions_fraud_features_test.go`.

### Two-hidden-layer fraud model

New `transactions_fraud.go` training uses 11 inputs → 64 ReLU neurons → 32 ReLU
neurons → 1 sigmoid output. Set widths with `--hidden-neurons` and
`--hidden-neurons-2` (both positive). Hidden weights use seeded He initialization;
BCE backpropagation uses ReLU derivative 1 above zero and 0 otherwise.

The saved `hiddenNeurons2` configuration and three parameter matrices identify
this architecture. Omitted/zero second-layer width preserves the shared library's
legacy single sigmoid hidden layer, including Iris and compatible saved models.
The historical standalone trainer is unchanged. Retrain to use the new architecture.


### Weighted BCE and validation thresholds

The active `transactions_fraud.go` trainer defaults to weighted BCE with positive
weight `training legitimate rows / training fraud rows`. Only loaded training rows
count (including the `--max-rows` limit); automatic weighting requires both classes.
The loss is `-mean(w*y*log(p) + (1-y)*log(1-p))`, averaged over rows rather than
summed weights. Fraud output gradients are multiplied by the same weight.

Use the automatic ratio, or supply a smaller positive weight to experiment:

```sh
go run transactions_fraud.go --mode train --positive-weight auto --model weighted_fraud.json
go run transactions_fraud.go --mode train --positive-weight 100 --model weighted_fraud_100.json
```

The resolved weight is printed and saved at `model.config.positive_weight`.
Older models without this field retain ordinary BCE. Epoch logs show
`weighted_bce`; final split results show ordinary `bce` alongside `weighted_bce`.
In saved split summaries, `loss` remains ordinary BCE and `weighted_bce` uses the
training weight for all three splits. Epoch metrics are measured after all updates in that epoch; final losses are
measured after training.

Each run prints a validation table of Threshold, Precision, Recall and F2 for
0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75 and 0.80, saved as
`summary.validation_thresholds`. The operating threshold maximizes validation F2 by
sorting scores and considering every distinct score with the `score >= threshold`
rule. Ties within `1e-12` prefer higher recall, then higher precision, then a lower
threshold. This takes O(N log N) time and O(N) extra memory. Validation must contain
at least one fraud case; tuning errors stop training before a model is saved.
Test metrics use the selected validation threshold. Selected thresholds are saved
without rounding and printed with 17 significant digits.

Remove `--threshold-step` from active-trainer commands: the flag and
`summary.threshold_step` have been removed. Older saved models still use their
saved thresholds unchanged; legacy JSON `threshold_step` fields are ignored.
The historical `transactions_fraud_nn.go` still supports its grid-search flag.

Larger positive weights can increase false positives. Compare lower weights on
validation data; weighted sigmoid scores are not necessarily calibrated fraud
probabilities. The historical standalone trainer is unchanged.

### Mini-batch training

The active fraud CLI defaults to `--batch-size 512 --epochs 1 --learning-rate 0.001`.
Each epoch shuffles paired training rows using the configured seed and updates
weights after each batch, including the shorter final batch. For 4,445,835 rows
this means 8,684 updates, with 139 rows in the final batch. Positive-class weighting
is still calculated from the entire loaded training split.

Epoch logs now report training and validation BCE, weighted BCE and AP after the
updates. Predictions for these evaluations are computed in chunks to bound hidden
activation memory. Inspect the first epoch before increasing `--epochs`; old
full-batch epoch counts are not comparable. Final threshold selection is unchanged.
The saved `batch_size` configuration records the training setting; omitted/zero
values retain full-batch behavior for shared-library callers and older models.

```sh
go run transactions_fraud.go --mode train --batch-size 512 --epochs 1 --learning-rate 0.001 --model minibatch_fraud.json
```

Final fraud training output also reports the mean, median, 90th and 99th
percentiles of predicted fraud probabilities, grouped by the true legitimate/fraud
label, for train, validation and test. These threshold-independent summaries are
saved under each split's `probability_stats`. Percentiles use linear interpolation
at `(count-1)*quantile`. Missing classes print N/A and serialize as null.
The reusable helper is `irisnn.SummarizeClassProbabilities`; it validates inputs
and leaves the supplied probabilities and labels unchanged. Summaries are computed
only at final evaluation, not on every epoch.

### Optimizer selection

The active fraud CLI defaults to Adam. Switch to the previous mini-batch gradient
descent with `--optimizer sgd`. Both use `--learning-rate` and the same averaged,
positive-class-weighted gradients. Adam uses beta1=0.9, beta2=0.999 and epsilon=1e-8,
with one shared timestep per batch and separate moments for every weight and bias.

```sh
go run transactions_fraud.go --optimizer adam --epochs 1 --model adam_fraud.json
go run transactions_fraud.go --optimizer sgd --epochs 1 --model sgd_fraud.json
```

The optimizer choice is saved in `model.config.optimizer`. Missing values preserve
SGD for older models and library callers. Adam state persists across training calls
on the same network, but moments and timestep are not serialized: loading preserves
predictions and starts fresh optimizer state for further training. Compare validation
AP and BCE before increasing epochs; Adam is not guaranteed to improve results.
