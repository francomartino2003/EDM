"""
Main script to run complete experiments
Simple execution - everything runs sequentially from preprocessing to evaluation
"""
from config import Config
from train import Trainer
from evaluate import Evaluator
import os


def main():
    """
    Main experiment execution
    Creates experiment folder structure and runs complete pipeline
    """
    
    # Configuration
    config = Config()  # experiment_name is defined in config.py
    
    # Create experiment-specific folders
    experiment_dir = os.path.join(config.results_dir, config.experiment_name)
    config.model_dir = os.path.join(experiment_dir, "models")
    config.results_dir = os.path.join(experiment_dir, "results")
    config.logs_dir = os.path.join(experiment_dir, "logs")
    
    # Create directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    print("="*80)
    print(f"STARTING EXPERIMENT: {config.experiment_name}")
    print("="*80)
    
    # Print configuration
    print("\n" + "-"*80)
    print("CONFIGURATION")
    print("-"*80)
    print(f"Experiment name:      {config.experiment_name}")
    print(f"Results directory:    {experiment_dir}")
    print()
    print("Preprocessing:")
    print(f"  Chunk length:       {config.chunk_length}")
    print(f"  Stride:             {config.stride}")
    print(f"  Normalize:          {config.normalize}")
    print()
    print("Architecture:")
    print(f"  Channels:           {config.channels}")
    print(f"  Dilations:          {config.dilations}")
    print(f"  Kernel size:        {config.kernel_size}")
    print(f"  Dropout:            {config.dropout}")
    print(f"  Receptive field:    {config.receptive_field}")
    print()
    print("Training:")
    print(f"  Batch size:         {config.batch_size}")
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Weight decay:       {config.weight_decay}")
    print(f"  Num epochs:         {config.num_epochs}")
    print(f"  Early stopping:     {config.early_stopping_patience}")
    print()
    print("Target segments:")
    for i, seg in enumerate(config.target_segments):
        print(f"  {i}: {seg}")
    print(f"  {len(config.target_segments)}: {config.other_segment}")
    
    # Save configuration
    config_path = os.path.join(experiment_dir, f"{config.experiment_name}_config.json")
    config.save_config(config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    # STEP 1: TRAINING
    print("\n" + "="*80)
    print("STEP 1: TRAINING")
    print("="*80)
    
    trainer = Trainer(config)
    trainer.train()
    
    print(f"\nTraining completed. Best validation loss: {trainer.best_val_loss:.4f}")
    
    # STEP 2: EVALUATION
    print("\n" + "="*80)
    print("STEP 2: EVALUATION")
    print("="*80)
    
    # Use the model directly from trainer
    evaluator = Evaluator(config, trainer.model, trainer.preprocessor)
    metrics = evaluator.evaluate()
    
    print(f"\nEvaluation completed")
    
    # STEP 3: GENERATE VISUALIZATION
    print("\n" + "="*80)
    print("STEP 3: GENERATE VISUALIZATION")
    print("="*80)
    
    try:
        from visualize_predictions import visualize_predictions
        csv_file = "../Data/Option 2/Test/Normal/0c78b343ab784034ac1d604940910881.csv"
        model_path = os.path.join(config.model_dir, f"{config.experiment_name}_best_model.pth")
        config_path = os.path.join(experiment_dir, f"{config.experiment_name}_config.json")
        output_dir = os.path.join(config.results_dir, "visualizations")
        
        print(f"Generating visualization for: {csv_file}")
        visualize_predictions(csv_file, model_path, config_path, output_dir)
        print(f"Visualization saved to: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nExperiment folder: {experiment_dir}")
    print(f"\nContents:")
    print(f"  - {config.experiment_name}_config.json")
    print(f"  - models/")
    print(f"  - results/{config.experiment_name}_history.json")
    print(f"  - results/{config.experiment_name}_metrics.json")
    print(f"  - results/visualizations/")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()