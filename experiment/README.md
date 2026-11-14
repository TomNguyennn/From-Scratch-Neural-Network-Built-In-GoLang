# PyTorch fraud experiments

From the repository root:

```sh
experiment/.venv/bin/python -m pip install -r experiment/requirements.txt
experiment/.venv/bin/python experiment/neural-net.py
```

Open `fraud_experiments.ipynb` in VS Code or Jupyter using the interpreter in
`experiment/.venv`. Cells expose loading, preprocessing, scaling, features,
model/optimizer creation, training, validation, evaluation and recording separately.
Call `train_epoch` again to continue training. Changing preprocessing requires
rebuilding downstream features. Training transformations never fit on evaluation data.

Defaults: 200k/50k/50k stratified train/validation/test samples, three epochs,
64/32 hidden neurons, Adam at 0.001, batch size 256, seed 42, CPU float32.
Use `--help` for configurable settings. Sample size 0 means the entire split.
Sampling reads labels and then streams selected records; it still scans the CSVs.

```sh
experiment/.venv/bin/python experiment/neural-net.py --train-size 50000 --validation-size 10000 --test-size 10000 --epochs 1
experiment/.venv/bin/python -m unittest discover -s experiment -p 'test_*.py'
```

Each run writes history.csv, metrics.json, validation_thresholds.csv,
sample_indices.npz (zero-based CSV data row indices), config.json and model.pt
under a unique runs directory. `load_checkpoint` returns the CPU model, optimizer,
scaler, threshold and configuration; its optimizer can continue training.
Notebook edits to architecture/features must also update the checkpoint contract.

The replica follows the active balance-free Go model. Reported BCE clips at 1e-9;
training uses stable BCEWithLogitsLoss. Thresholds maximize validation F2 with the
Go tie-breaking rule. Test labels never select a threshold. AP is non-interpolated.
The historical L2 flag was unused, so this replica uses no weight decay.
Initialization distributions match Go, but random streams, float32 arithmetic and
backend operations differ: scores and training trajectories need not match.
Small samples have few fraud cases; inspect class counts before comparing metrics.
No Go weight conversion or external tracking service is included.

`requirements-tested.txt` records the complete environment used for verification.
