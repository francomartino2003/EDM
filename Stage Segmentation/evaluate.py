"""
Evaluation of the model on test data
Threshold-independent metrics (AUC, etc.)
"""
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)

from config import Config
from model import create_model
from preprocessing import DataPreprocessor


class Evaluator:
    """Evaluador del modelo"""
    
    def __init__(self, config: Config, model, preprocessor=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use provided model
        self.model = model
        
        # Preprocesar datos de test (use provided preprocessor or create new one)
        if preprocessor is None:
            preprocessor = DataPreprocessor(config)
        self.X_test, self.y_test, stats = preprocessor.preprocess_test(config.test_path)
        self.stats_test = stats
        
        print(f"Using model from trainer")
        print(f"Datos de test: {len(self.X_test)} series")
    
    def _load_model(self, model_path: str):
        """Cargar modelo entrenado"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Recuperar config si existe
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            # Crear nuevo config
            new_config = Config()
            # Asignar valores
            for key, value in config_dict.items():
                setattr(new_config, key, value)
            self.config = new_config
        
        # Crear modelo
        model = create_model(self.config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def evaluate_series(self, X_series, y_series):
        """
        Evaluar una serie completa
        Args:
            X_series: (length, 2)
            y_series: (length,)
        Returns:
            predictions: (length,)
            probs: (num_classes, length)
        """
        # Convertir a tensor
        x = torch.FloatTensor(X_series).transpose(0, 1).unsqueeze(0).to(self.device)  # (1, 2, length)
        
        # Predecir
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        return predictions[0].cpu().numpy(), probs[0].cpu().numpy()
    
    def evaluate(self):
        """Evaluate on all test series"""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        all_predictions = []
        all_probabilities = []
        all_ground_truth = []
        
        # Evaluate series by series
        for X_series, y_series in tqdm(zip(self.X_test, self.y_test), total=len(self.X_test)):
            predictions, probs = self.evaluate_series(X_series, y_series)
            
            all_predictions.append(predictions)
            all_probabilities.append(probs.T)  # Transpose to (length, num_classes)
            all_ground_truth.append(y_series)
        
        # Concatenate
        all_predictions = np.concatenate(all_predictions)
        all_probabilities = np.concatenate(all_probabilities)  # (total_samples, num_classes)
        all_ground_truth = np.concatenate(all_ground_truth)
        
        # Compute dataset statistics
        dataset_stats = self._compute_dataset_stats(all_ground_truth)
        
        # Metrics (threshold-independent)
        metrics = self._compute_metrics(all_predictions, all_probabilities, all_ground_truth)
        
        # Add dataset stats to metrics
        metrics['dataset_info'] = dataset_stats
        
        # Print results
        self._print_results(metrics)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _compute_dataset_stats(self, ground_truth):
        """Compute dataset statistics"""
        from model import count_parameters
        
        class_names = self.config.target_segments + [self.config.other_segment]
        
        # Class distribution
        unique_classes, class_counts = np.unique(ground_truth, return_counts=True)
        total_samples = len(ground_truth)
        
        class_distribution = {}
        for cls_idx, count in zip(unique_classes, class_counts):
            class_name = class_names[cls_idx]
            percentage = (count / total_samples) * 100
            class_distribution[class_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Model parameters
        num_params = count_parameters(self.model)
        receptive_field = self.model.get_receptive_field()
        
        # Dataset size
        num_series = len(self.X_test)
        total_samples = len(ground_truth)
        
        return {
            'num_series': num_series,
            'total_samples': int(total_samples),
            'num_classes': self.config.num_classes,
            'class_distribution': class_distribution,
            'model_parameters': {
                'total_parameters': num_params,
                'receptive_field': receptive_field,
                'channels': self.config.channels,
                'dilations': self.config.dilations,
                'kernel_size': self.config.kernel_size,
                'dropout': self.config.dropout
            },
            'data_info': {
                'train_path': self.config.train_path,
                'test_path': self.config.test_path,
                'chunk_length': self.config.chunk_length,
                'stride': self.config.stride,
                'normalize': self.config.normalize
            }
        }
    
    def _compute_metrics(self, predictions, probabilities, ground_truth):
        """Compute metrics (threshold-independent)"""
        # Accuracy (threshold-dependent, but keep for reference)
        accuracy = accuracy_score(ground_truth, predictions)
        
        # One-hot encode ground truth for AUC
        y_true_one_hot = np.eye(self.config.num_classes)[ground_truth]
        
        # ROC AUC - macro average (threshold-independent)
        try:
            roc_auc_macro = roc_auc_score(
                y_true_one_hot, probabilities, 
                average='macro', multi_class='ovr'
            )
        except ValueError:
            roc_auc_macro = 0.0
        
        # ROC AUC - weighted average
        try:
            roc_auc_weighted = roc_auc_score(
                y_true_one_hot, probabilities, 
                average='weighted', multi_class='ovr'
            )
        except ValueError:
            roc_auc_weighted = 0.0
        
        # Per-class ROC AUC
        roc_auc_per_class = []
        for i in range(self.config.num_classes):
            try:
                auc = roc_auc_score(
                    y_true_one_hot[:, i], probabilities[:, i]
                )
                roc_auc_per_class.append(auc)
            except ValueError:
                roc_auc_per_class.append(0.0)
        
        # Average Precision (macro)
        try:
            avg_precision_macro = average_precision_score(
                y_true_one_hot, probabilities, 
                average='macro'
            )
        except ValueError:
            avg_precision_macro = 0.0
        
        # Average Precision (weighted)
        try:
            avg_precision_weighted = average_precision_score(
                y_true_one_hot, probabilities, 
                average='weighted'
            )
        except ValueError:
            avg_precision_weighted = 0.0
        
        # Per-class average precision
        avg_precision_per_class = []
        for i in range(self.config.num_classes):
            try:
                ap = average_precision_score(
                    y_true_one_hot[:, i], probabilities[:, i]
                )
                avg_precision_per_class.append(ap)
            except ValueError:
                avg_precision_per_class.append(0.0)
        
        # Per-class threshold-dependent metrics (for reference)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, 
            labels=range(self.config.num_classes),
            average=None,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Class names
        class_names = self.config.target_segments + [self.config.other_segment]
        
        metrics = {
            # Threshold-independent metrics
            'roc_auc_macro': float(roc_auc_macro),
            'roc_auc_weighted': float(roc_auc_weighted),
            'avg_precision_macro': float(avg_precision_macro),
            'avg_precision_weighted': float(avg_precision_weighted),
            # Threshold-dependent metrics (for reference)
            'accuracy': float(accuracy),
            # Per-class metrics
            'per_class': {
                class_names[i]: {
                    # Threshold-independent
                    'roc_auc': float(roc_auc_per_class[i]),
                    'avg_precision': float(avg_precision_per_class[i]),
                    # Threshold-dependent
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(self.config.num_classes)
            },
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def _print_results(self, metrics):
        """Print results"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        # Dataset info
        if 'dataset_info' in metrics:
            ds_info = metrics['dataset_info']
            print(f"\nDATASET INFO:")
            print(f"  Number of series:   {ds_info['num_series']}")
            print(f"  Total samples:      {ds_info['total_samples']:,}")
            print(f"  Number of classes:  {ds_info['num_classes']}")
            
            print(f"\nCLASS DISTRIBUTION:")
            for class_name, dist in ds_info['class_distribution'].items():
                print(f"  {class_name:20s}: {dist['count']:>6} ({dist['percentage']:>5.1f}%)")
            
            print(f"\nMODEL PARAMETERS:")
            model_params = ds_info['model_parameters']
            print(f"  Total parameters:   {model_params['total_parameters']:,}")
            print(f"  Receptive field:    {model_params['receptive_field']}")
            print(f"  Channels:           {model_params['channels']}")
            print(f"  Dilations:          {model_params['dilations']}")
            print(f"  Kernel size:        {model_params['kernel_size']}")
            print(f"  Dropout:            {model_params['dropout']}")
        
        # Threshold-independent metrics
        print(f"\nTHRESHOLD-INDEPENDENT METRICS:")
        if 'roc_auc_macro' in metrics:
            print(f"  ROC AUC (Macro):      {metrics['roc_auc_macro']:.4f}")
            print(f"  ROC AUC (Weighted):   {metrics['roc_auc_weighted']:.4f}")
            print(f"  Avg Precision (Macro): {metrics['avg_precision_macro']:.4f}")
            print(f"  Avg Precision (Weight): {metrics['avg_precision_weighted']:.4f}")
        
        # Threshold-dependent metrics (for reference)
        print(f"\nTHRESHOLD-DEPENDENT METRICS (for reference):")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        
        print(f"\nPER-CLASS METRICS:")
        class_names = self.config.target_segments + [self.config.other_segment]
        
        for class_name in class_names:
            class_metrics = metrics['per_class'][class_name]
            print(f"\n  {class_name}:")
            if 'roc_auc' in class_metrics:
                print(f"    ROC AUC:        {class_metrics['roc_auc']:.4f}")
                print(f"    Avg Precision:  {class_metrics['avg_precision']:.4f}")
            print(f"    Precision:      {class_metrics['precision']:.4f}")
            print(f"    Recall:         {class_metrics['recall']:.4f}")
            print(f"    F1:             {class_metrics['f1']:.4f}")
            print(f"    Support:        {class_metrics['support']}")
    
    def _save_results(self, metrics):
        """Guardar resultados"""
        import json
        
        results_dir = self.config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, f"{self.config.experiment_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResultados guardados en: {metrics_path}")


def main():
    """Main evaluation"""
    config = Config()
    config.experiment_name = "exp_001"
    model_path = os.path.join(config.model_dir, f"{config.experiment_name}_best_model.pth")
    
    evaluator = Evaluator(config, model_path)
    metrics = evaluator.evaluate()
    
    print("\n" + "="*60)
    print("EVALUACIÓN COMPLETADA")
    print("="*60)


if __name__ == "__main__":
    main()
