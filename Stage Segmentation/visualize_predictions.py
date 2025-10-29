"""
Visualize model predictions - simplified version
Shows only probabilities with stage boundaries
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import Config
from model import create_model
from preprocessing import DataPreprocessor

def visualize_predictions(csv_file, model_path, config_path, output_dir="predictions_vis"):
    # Load config
    config = Config.load_config(config_path)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = create_model(config).to('cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    from sklearn.preprocessing import StandardScaler
    import glob
    train_csvs = glob.glob(str(Path(config.train_path) / "**" / "*.csv"), recursive=True)
    all_data = []
    for f in train_csvs:
        df_temp = pd.read_csv(f)
        all_data.append(df_temp[['Voltage', 'Z']].values)
    all_data = np.concatenate(all_data, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_data)
    
    # Load test data
    df = pd.read_csv(csv_file)
    features = df[['Voltage', 'Z']].values
    true_segments = df['Segment'].values
    features_scaled = scaler.transform(features).astype(np.float32)
    
    # Predict
    x = torch.FloatTensor(features_scaled).transpose(0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    probs = probs[0].cpu().numpy()
    
    # Smooth probabilities with moving average
    def moving_average(data, window_size=10):
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed[i] = np.mean(data[start:end], axis=0)
        return smoothed
    
    # Apply smoothing to each class
    probs_smoothed = np.array([moving_average(probs[i], window_size=10) for i in range(len(probs))])
    probs = probs_smoothed
    
    # Map segments
    class_names = config.target_segments + [config.other_segment]
    segment_to_idx = {name: i for i, name in enumerate(class_names)}
    true_indices = np.array([segment_to_idx.get(s, len(class_names)-1) for s in true_segments])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))
    time = np.arange(len(df))
    
    # Plot probabilities
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        ax.plot(time, probs[i], color=color, linewidth=2, label=class_name, alpha=0.8)
    
    # Add vertical lines for boundaries
    segment_changes = np.where(np.diff(true_indices) != 0)[0]
    for change_idx in segment_changes:
        ax.axvline(x=change_idx, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
    
    # Add shaded regions
    current_seg = true_indices[0]
    start_idx = 0
    segment_colors = ['#e8eaf6', '#e6f4ea', '#fff4e5', '#fde8e8', '#f3e8ff']
    
    for i, seg_idx in enumerate(true_indices[1:], 1):
        if seg_idx != current_seg:
            ax.axvspan(start_idx, i, alpha=0.15, color=segment_colors[current_seg])
            mid_idx = (start_idx + i) // 2
            ax.text(mid_idx, 0.95, class_names[current_seg], ha='center', va='top', 
                   fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.7))
            start_idx = i
            current_seg = seg_idx
    
    # Last segment
    if start_idx < len(true_indices):
        ax.axvspan(start_idx, len(true_indices), alpha=0.15, color=segment_colors[current_seg])
        mid_idx = (start_idx + len(true_indices)) // 2
        ax.text(mid_idx, 0.95, class_names[current_seg], ha='center', va='top', 
               fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.028', 
               facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Model Predictions: {Path(csv_file).name}', fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_file = Path(output_dir) / f"{Path(csv_file).stem}_predictions.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    csv_file = "../Data/Option 2/Test/Normal/0c78b343ab784034ac1d604940910881.csv"
    model_path = "results/exp_001/models/exp_001_best_model.pth"
    config_path = "results/exp_001/exp_001_config.json"
    
    visualize_predictions(csv_file, model_path, config_path)