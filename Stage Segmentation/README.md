# Stage Segmentation

Temporal segmentation system for EDM drilling stages using Temporal Convolutional Networks (TCN) with dilated convolutions.

## Overview

This project implements a deep learning pipeline for temporal segmentation of EDM drilling stages. The model takes Voltage and Z time series as input and predicts the stage (Touching, Body Drilling, Break-through, Free Falling, or others) at each time step.

## Structure

```
Stage Segmentation/
├── config.py              # Experiment configuration (ALL hyperparameters)
├── preprocessing.py       # Data preprocessing and chunking
├── model.py               # TCN architecture with dilated convolutions
├── train.py               # Training loop with validation split
├── evaluate.py            # Evaluation with threshold-independent metrics
├── visualize_predictions.py  # Generate prediction visualizations
├── run_experiment.py      # Main script to run complete pipeline
├── results/               # Experiment results (config, history, metrics, visualizations)
│   └── exp_XXX/
│       ├── models/        # Saved model checkpoints
│       ├── results/       # JSON results
│       └── logs/          # Training logs
```

## Quick Start

### Setup

1. Activate conda environment:
```bash
conda activate edm_plotting
```

2. Configure hyperparameters in `config.py`:
```python
experiment_name = "exp_001"
chunk_length = 600
stride = 2
# ... etc
```

3. Run complete experiment:
```bash
python run_experiment.py
```

The script will:
- Load and preprocess data
- Split series into train/validation (10% validation)
- Create chunks with overlap
- Train the model
- Evaluate on test set
- Generate visualization for a test sample
- Save all results

## Key Features

### Data Preprocessing
- **No data leakage**: Series are split BEFORE chunking, not after
- Standardization of features (Voltage, Z)
- Configurable chunking (length, stride)
- Automatic class mapping and statistics

### Model Architecture
- **TCN (Temporal Convolutional Network)**: Non-causal with dilated convolutions
- **Receptive Field**: Automatically calculated based on dilations and kernel size
- **Length preservation**: Output sequence length = input length (no cropping)
- Batch normalization and dropout for regularization

### Training
- **Focal Loss**: Handles class imbalance
- **Class weights**: Inverse frequency weighting
- **80/10% split**: Train/validation split at SERIES level
- **Early stopping**: Based on validation loss
- **Learning rate scheduler**: ReduceLROnPlateau
- **Gradient clipping**: Prevents exploding gradients
- **CUDA detection**: Automatically uses GPU if available

### Evaluation
- **Threshold-independent metrics**: ROC AUC, Average Precision
- **Complete series evaluation**: Test data processed without chunking
- **Multiple metrics**: Accuracy, Precision, Recall, F1 (per-class and global)
- **Dataset statistics**: Model parameters, class distribution, data info

### Visualization
- Automatic generation for test sample
- Smoothed probability curves
- True stage boundaries (vertical lines, shaded backgrounds)
- Saved to results folder

## Hyperparameters

All hyperparameters are defined in `config.py`:

### Data & Preprocessing
| Parameter | Description | Default | Example |
|---|---|---|---|
| `chunk_length` | Length of training chunks | 600 | 512, 1024 |
| `stride` | Stride between chunks | 2 | 1, 25 |
| `normalize` | Standardize features | True | True/False |
| `target_segments` | Classes to predict | [Touching, Body Drilling, Break-through, Free Falling] | - |
| `other_segment` | Name for grouped class | "others" | "misc" |

### Architecture
| Parameter | Description | Default | Example |
|---|---|---|---|
| `channels` | Channels per layer | [64, 128, 256] | [32, 64, 128, 256] |
| `dilations` | Dilation rates | [1, 2, 8] | [1, 2, 4, 8] |
| `kernel_size` | Convolution kernel size | 5 | 3, 5, 7 |
| `dropout` | Dropout rate | 0.25 | 0.1, 0.3 |

### Training
| Parameter | Description | Default | Example |
|---|---|---|---|
| `batch_size` | Batch size | 256 | 32, 64, 128 |
| `learning_rate` | Initial LR | 1e-4 | 1e-3, 5e-4 |
| `weight_decay` | L2 regularization | 1e-3 | 1e-4, 1e-5 |
| `num_epochs` | Maximum epochs | 50 | 100 |
| `early_stopping_patience` | Early stop patience | 10 | 15, 20 |
| `scheduler_factor` | LR reduction factor | 0.9 | 0.5, 0.8 |
| `scheduler_patience` | LR scheduler patience | 5 | 3, 10 |

### Paths
- `train_path`: Path to training data (`../Data/Option 2/Train`)
- `test_path`: Path to test data (`../Data/Option 2/Test`)

## Data Structure

Expected data format:
```
Data/Option 2/
├── Train/
│   ├── Normal/
│   ├── NPT/
│   ├── OD/
│   ├── MH/
│   └── ...
└── Test/
    ├── Normal/
    ├── NPT/
    ├── OD/
    └── ...
```

Each CSV file should contain:
- `Voltage`: Voltage measurements
- `Z`: Depth measurements
- `Segment`: Stage labels

## Experiment Results

### Summary of Completed Experiments

| Exp | Chunk | Stride | Layers | RF | Params | ROC AUC (Macro) | Accuracy | Val Split |
|---|---|---|---|---|---|---|---|---|
| exp_001 | 512 | 25 | 3 | 31 | 175K | - | - | Random chunks |
| exp_002 | 300 | 1 | 4 | 31 | 175K | 0.8670 | 0.6480 | Random chunks |
| exp_003 | 600 | 3 | 4 | 31 | 175K | 0.8742 | 0.6659 | Fixed (20%) |
| exp_004 | 600 | 2 | 3 | 23 | 126K | 0.8726 | 0.6370 | Fixed (10%) |
| exp_005 | 650 | 2 | 3 | 45 | 208K | 0.8690 | 0.6694 | Fixed (10%) |
| exp_006 | 600 | 2 | 3 | 45 | 208K | 0.8920 | 0.6755 | Fixed (10%) |
| exp_007 | 600 | 2 | 5 | 125 | 619K | 0.8955 | 0.7071 | Fixed (10%) |

**Best model**: exp_007 (ROC AUC: 0.8955, Accuracy: 0.7071)

## Output Files

### For Each Experiment (`results/exp_XXX/`)

#### Models (`models/`)
- `{exp}_best_model.pth`: Best model (lowest validation loss)
- `{exp}_final_model.pth`: Model from last epoch

#### Results (`results/`)
- `{exp}_config.json`: Complete configuration
- `{exp}_history.json`: Training history (loss, accuracy per epoch)
- `{exp}_metrics.json`: Evaluation metrics on test set
- `visualizations/`: Prediction plots

## Metrics Explained

### Threshold-Independent (Levelled as most important)
- **ROC AUC**: Area under ROC curve (higher = better, max = 1.0)
- **Average Precision**: Area under PR curve (higher = better, max = 1.0)
- Both metrics don't depend on threshold choice

### Threshold-Dependent (Reference)
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Per-class and averaged metrics

## Troubleshooting

### Kernel Size Issues
- Use **odd kernel sizes** (3, 5, 7) for cleaner length preservation
- Even sizes (4, 6, 8) work but may have edge cases

### Memory Issues
- Reduce `batch_size`
- Reduce `chunk_length`
- Reduce model size (`channels`)

### Overfitting
- Increase `dropout`
- Increase `weight_decay`
- Increase `early_stopping_patience`

### Poor Performance
- Increase `receptive_field` (more layers/dilations)
- Increase model capacity (`channels`)
- Experiment with `chunk_length` and `stride`

## Notes

- All code, comments, and documentation are in English
- The model outputs **logits** (not probabilities) during training
- CUDA device is automatically detected and used if available
- Results are saved with dataset statistics and model parameters
