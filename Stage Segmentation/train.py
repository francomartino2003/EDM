"""
Training script
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

from config import Config
from model import create_model, FocalLoss, count_parameters
from preprocessing import DataPreprocessor


class ChunkDataset(Dataset):
    """Dataset for temporal series chunks"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  # (n_chunks, length, 2)
        self.y = torch.LongTensor(y)   # (n_chunks, length)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert to (2, length) for model input
        x = self.X[idx].transpose(0, 1)  # (2, length)
        y = self.y[idx]  # (length)
        return x, y


class Trainer:
    """Model trainer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Using device: CPU (CUDA not available)")
        
        # Crear directorios
        self._create_directories()
        
        # Guardar config
        config.save_config(os.path.join(config.results_dir, f"{config.experiment_name}_config.json"))
        
        # Cargar y preprocesar datos
        print("\n" + "="*60)
        print("PREPROCESAMIENTO")
        print("="*60)
        
        self.preprocessor = DataPreprocessor(config)
        
        X_train, y_train, stats_train = self.preprocessor.preprocess_train(config.train_path)
        self.stats_train = stats_train
        
        # Crear datasets y dataloaders
        # Get validation chunks from preprocessor (already split at series level to avoid data leakage)
        X_val, y_val = self.preprocessor.get_validation_chunks()
        
        train_dataset = ChunkDataset(X_train, y_train)
        val_dataset = ChunkDataset(X_val, y_val)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Crear modelo
        print("\n" + "="*60)
        print("MODEL")
        print("="*60)
        
        self.model = create_model(config).to(self.device)
        num_params = count_parameters(self.model)
        print(f"Model created: {num_params:,} parameters")
        print(f"Receptive field: {self.model.get_receptive_field()}")
        
        # Loss y optimizador
        # Calcular class weights para balancear
        class_counts = np.array([stats_train['class_counts'][name]['count'] 
                                for name in config.target_segments + [config.other_segment]])
        
        # Inverse frequency weighting
        total = class_counts.sum()
        class_weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
        class_weights = class_weights / class_weights.sum()
        
        print(f"\nClass weights:")
        for name, weight in zip(config.target_segments + [config.other_segment], class_weights):
            print(f"  {name}: {weight:.4f}")
        
        # Usar Focal Loss
        self.criterion = FocalLoss(gamma=2.0, class_weights=class_weights.to(self.device))
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _create_directories(self):
        """Crear directorios necesarios"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Entrenar una época"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Train")
        
        for batch_idx, (x, y) in enumerate(pbar):
            # Activar debug para época 2, batch 0
            if epoch == 2 and batch_idx == 0:
                self.model.debug = True
                print(f"\nDEBUG: Epoch {epoch}, Batch {batch_idx}")
            elif batch_idx == 1:
                self.model.debug = False
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(x)  # (batch, num_classes, length)
            
            # Loss
            loss = self.criterion(logits, y)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == y).sum().item()
            total_correct += correct
            total_samples += predictions.numel()
            
            # Update bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = total_correct / total_samples
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward
                logits = self.model(x)
                
                # Loss
                loss = self.criterion(logits, y)
                
                # Metrics
                total_loss += loss.item()
                
                # Accuracy
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == y).sum().item()
                total_correct += correct
                total_samples += predictions.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(self):
        """Loop principal de entrenamiento"""
        print("\n" + "="*60)
        print("ENTRENAMIENTO")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler step (use validation loss)
            self.scheduler.step(val_loss)
            
            # Print summary
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print(f"  ✓ New best model (val loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Save history
        self.save_history()
        
        training_time = time.time() - start_time
        print(f"\nTiempo de entrenamiento: {training_time/60:.2f} minutos")
    
    def save_model(self, filename):
        """Guardar modelo"""
        path = os.path.join(self.config.model_dir, f"{self.config.experiment_name}_{filename}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'stats': self.stats_train
        }, path)
    
    def save_history(self):
        """Save training history"""
        path = os.path.join(self.config.results_dir, f"{self.config.experiment_name}_history.json")
        
        history_dict = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_acc': self.history['train_acc'],
            'val_acc': self.history['val_acc'],
            'lr': self.history['lr'],
            'best_val_loss': self.best_val_loss
        }
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)


# Main function removed - use run_experiment.py instead
# This ensures all hyperparameters come from config.py only
