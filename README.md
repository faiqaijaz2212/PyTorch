# Fashion-MNIST ANN with Optuna Hyperparameter Optimization

PyTorch implementation of a feedforward neural network for Fashion-MNIST classification, with automated hyperparameter tuning via Optuna.

## Overview

- **Dataset**: Fashion-MNIST (60,000 training, 10,000 test images, 28×28 grayscale, 10 classes)
- **Model**: Fully connected ANN with configurable depth/width, BatchNorm, ReLU, Dropout
- **Optimization**: Optuna tunes layers, neurons, epochs, LR, dropout, batch size, optimizer, weight decay
- **Hardware**: GPU-enabled (CUDA) with `pin_memory=True` for faster data loading

## Requirements

```bash
torch
torchvision
pandas
scikit-learn
optuna
matplotlib
```

You can install via pip (e.g., in a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install torch torchvision pandas scikit-learn optuna matplotlib
```

## Quick Start

1. **Prepare data** (download `fashion-mnist_train.csv` into this repo)
2. **Open the notebook**: `ann_fashion_pytorch_gpu_optuna.ipynb`
3. **Run cells** in order:
   - Imports and device check
   - Load and split/normalize data
   - Define `CustomDataset` and create `DataLoader`s
   - Define `MyNN` model class
   - Define `objective(trial)` for Optuna
   - Create study and run optimization:

```python
import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # increase n_trials for deeper search
```

4. **Results**:
   - Best trial: `study.best_trial.params`
   - Best accuracy: `study.best_value`
   - Full study: `study.trials`

## Notebook Structure

| Cell | Purpose |
|------|---------|
| 0–3 | Imports, seed, GPU detection |
| 4–6 | Load CSV, visualize samples |
| 7–10 | Train/test split, normalize |
| 11–13 | `CustomDataset`, `DataLoader`s |
| 14 | Model class `MyNN` |
| 15 | `objective(trial)` – Optuna target |
| 16–17 | Create and run Optuna study |

## Hyperparameters Searched

| Parameter | Search space |
|-----------|--------------|
| `num_hidden_layers` | 1–5 (int) |
| `neurons_per_layer` | 8–128 step 8 (int) |
| `epochs` | 10–50 step 10 (int) |
| `learning_rate` | 1e-5–1e-1 (log) |
| `dropout_rate` | 0.1–0.5 step 0.1 |
| `batch_size` | [16, 32, 64, 128] |
| `optimizer` | Adam, SGD, RMSprop |
| `weight_decay` | 1e-5–1e-3 (log) |

## Tips

- Increase `n_trials` for better results (e.g., 50–100)
- Enable Optuna’s `show_progress_bar=True` to monitor
- Use `study.trials_dataframe()` to export results
- To persist studies, pass a `storage` URI to `create_study`
- Consider pruning (`optuna.pruners.MedianPruner`) for early stopping on poor trials

## Example Best Result (from a 10‑trial run)

```
Best accuracy: 0.8911
Params:
{
  'num_hidden_layers': 5,
  'neurons_per_layer': 120,
  'epochs': 30,
  'learning_rate': 0.0405,
  'dropout_rate': 0.2,
  'batch_size': 64,
  'optimizer': 'SGD',
  'weight_decay': 2.40e-05
}
```

## License

Feel free to reuse/adapt for educational or research purposes.
