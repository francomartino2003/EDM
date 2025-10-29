# EDM

Deep learning system for studying and classifying EDM (Electrical Discharge Machining) drilling processes using temporal segmentation.

## Components

- **Data Analysis**: Statistical analysis and visualization tools
- **Stage Segmentation**: Temporal Convolutional Network (TCN) for stage segmentation

**Future**: Transfer learning models for additional EDM classification tasks

## 🚀 Quick Start

### Prerequisites

**IMPORTANT**: Before running any code, you need to add the `Data` folder with the following structure:

```
Data/
└── Option 2/
    ├── Train/
    │   ├── Normal/
    │   ├── NPT/
    │   ├── OD/
    │   └── ... (each containing CSV files)
    └── Test/
        ├── Normal/
        ├── NPT/
        ├── OD/
        └── ...
```

Each CSV file should contain: `Voltage`, `Z`, and `Segment` columns.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/francomartino2003/EDM.git
cd EDM
```

2. Create and activate conda environment:
```bash
conda create -n edm python=3.10
conda activate edm
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Optional: Generate Data Visualizations

Before running experiments, you can optionally generate visualization plots:

```bash
cd "Data Analysis"
python create_visualizations.py
```

This creates the `Data Viz` folder with plots for all time series. **Note**: This takes time.

### Running Stage Segmentation

```bash
cd "Stage Segmentation"
python run_experiment.py
```

Configure hyperparameters in `config.py` before running. See [Stage Segmentation/README.md](Stage%20Segmentation/README.md) for details.

## 📁 Project Structure

```
EDM/
├── Data/                          # (Excluded from repo) Raw CSV data
├── Data Analysis/
│   ├── README.md                  # Data analysis documentation
│   ├── create_visualizations.py   # Visualization generation
│   ├── data_analysis.py           # Statistical analysis
│   └── Data Viz/                  # (Excluded) Generated visualizations
├── Stage Segmentation/
│   ├── config.py                  # Experiment configuration
│   ├── preprocessing.py           # Data preprocessing
│   ├── model.py                   # TCN architecture
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── visualize_predictions.py   # Prediction visualization
│   ├── run_experiment.py          # Main execution script
│   ├── results/                   # Experiment results
│   └── README.md                  # Detailed documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```


## 🔧 Dependencies

See `requirements.txt` for complete list. Main dependencies:
- Python 3.10+
- PyTorch 2.0+
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn

## Author

**Franco Martino** - francomartino2003@gmail.com

Repository: https://github.com/francomartino2003/EDM
