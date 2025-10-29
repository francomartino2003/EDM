"""
Configuration for stage segmentation experiments
Allows easy hyperparameter variation between experiments
"""
from typing import List

class Config:
    """Experiment configuration"""
    
    # Experiment info
    experiment_name = "exp_007"
    seed = 42
    
    # Preprocessing
    chunk_length = 600  # Chunk length for training
    stride = 2  # Stride between chunks
    normalize = True  # Standardize series
    
    # Classes to predict
    target_segments = ["Touching", "Body Drilling", "Break-through", "Free Falling"]
    other_segment = "others"
    
    # Architecture hyperparameters
    channels = [64, 128, 128, 256, 256]
    dilations = [1, 2, 4, 8, 16]
    kernel_size = 5
    dropout = 0.25
    
    # Training hyperparameters
    batch_size = 256
    learning_rate = 1e-4
    weight_decay = 1e-3
    num_epochs = 50
    early_stopping_patience = 10
    
    # Learning rate scheduler
    scheduler_factor = 0.9  # Factor to reduce LR by
    scheduler_patience = 5  # Patience for LR reduction
    
    # Data paths
    data_path = "../Data/Option 2"
    train_path = "../Data/Option 2/Train"
    test_path = "../Data/Option 2/Test"
    
    # Output paths
    model_dir = "models"
    results_dir = "results"
    logs_dir = "logs"
    
    @property
    def num_classes(self) -> int:
        """Number of classes (target + others)"""
        return len(self.target_segments) + 1
    
    @property
    def receptive_field(self) -> int:
        """Calculate receptive field based on dilations and kernel_size"""
        rf = 1
        for dilation in self.dilations:
            rf += (self.kernel_size - 1) * dilation
        return rf
    
    def save_config(self, path: str):
        """Save configuration to file"""
        import json
        config_dict = {}
        for key in dir(self):
            if not key.startswith('_') and key not in ['save_config', 'load_config', 'num_classes', 'receptive_field']:
                value = getattr(self, key)
                if not callable(value):
                    config_dict[key] = value
        config_dict['num_classes'] = self.num_classes
        config_dict['receptive_field'] = self.receptive_field
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Create new instance
        config = cls()
        for key, value in config_dict.items():
            if key not in ['num_classes', 'receptive_field']:
                setattr(config, key, value)
        return config