"""
Preprocesamiento de series temporales EDM
"""
import numpy as np
import pandas as pd
import os
import glob
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from config import Config

class DataPreprocessor:
    """Data preprocessor for stage segmentation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.segment_mapping = self._create_segment_mapping()
        
    def _create_segment_mapping(self) -> Dict[str, int]:
        """Create mapping of segments to class indices"""
        mapping = {}
        for idx, segment in enumerate(self.config.target_segments):
            mapping[segment] = idx
        
        # Mapear "others" a la última clase
        mapping[self.config.other_segment] = len(self.config.target_segments)
        return mapping
    
    def load_raw_data(self, data_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Cargar datos brutos de CSV files
        Returns: Lista de tuplas (features, segments)
        features: (length, 2) - [Voltage, Z]
        segments: (length,) - segment labels
        """
        csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
        data = []
        
        print(f"Loading {len(csv_files)} files...")
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # Extraer features y segments
                features = df[['Voltage', 'Z']].values.astype(np.float32)
                segments = df['Segment'].values
                
                # Convertir segments a índices
                segment_indices = self._segments_to_indices(segments)
                
                data.append((features, segment_indices))
                
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(data)} files")
        return data
    
    def _segments_to_indices(self, segments: np.ndarray) -> np.ndarray:
        """Convertir labels de segments a índices de clases"""
        indices = np.zeros(len(segments), dtype=np.int32)
        
        for segment in np.unique(segments):
            if segment in self.config.target_segments:
                idx = self.config.target_segments.index(segment)
                indices[segments == segment] = idx
            else:
                # Mapear a "others"
                indices[segments == segment] = len(self.config.target_segments)
        
        return indices
    
    def create_chunks(self, features: np.ndarray, segments: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Crear chunks superpuestos de la serie"""
        chunks = []
        
        for start in range(0, len(features) - self.config.chunk_length + 1, self.config.stride):
            end = start + self.config.chunk_length
            
            chunk_features = features[start:end]
            chunk_segments = segments[start:end]
            
            chunks.append((chunk_features, chunk_segments))
        
        # Si la serie es muy corta, agregar el último chunk completo
        if len(chunks) == 0 and len(features) >= self.config.chunk_length:
            chunks.append((features[-self.config.chunk_length:], segments[-self.config.chunk_length:]))
        
        return chunks
    
    def preprocess_train(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocesar datos de entrenamiento
        Returns: (X_chunks, y_chunks, stats)
        """
        # Cargar datos
        data = self.load_raw_data(data_path)
        
        # IMPORTANT: Split series BEFORE creating chunks to avoid data leakage
        from sklearn.model_selection import train_test_split
        series_indices = list(range(len(data)))
        train_indices, val_indices = train_test_split(
            series_indices,
            test_size=0.1,  # 10% validation
            random_state=self.config.seed
        )
        
        train_series = [data[i] for i in train_indices]
        val_series = [data[i] for i in val_indices]
        
        # Store validation series for later (will be used separately in train.py)
        self._val_series = val_series
        
        print(f"\nSeries split: {len(train_series)} train, {len(val_series)} validation")
        
        # Crear chunks solo de series de entrenamiento
        print("Creating chunks...")
        all_chunks_features = []
        all_chunks_segments = []
        
        for features, segments in train_series:
            chunks = self.create_chunks(features, segments)
            
            for chunk_features, chunk_segments in chunks:
                all_chunks_features.append(chunk_features)
                all_chunks_segments.append(chunk_segments)
        
        # Convertir a arrays numpy
        X = np.array(all_chunks_features)  # (n_chunks, chunk_length, 2)
        y = np.array(all_chunks_segments)  # (n_chunks, chunk_length)
        
        # Estandarizar
        if self.config.normalize:
            print("Standardizing features...")
            # Reshape for fit: (n_chunks * chunk_length, 2)
            X_reshaped = X.reshape(-1, 2)
            self.scaler.fit(X_reshaped)
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(X.shape).astype(np.float32)
        
        # Estadísticas
        stats = self._compute_stats(y, "Train")
        
        print(f"\nDataset de entrenamiento:")
        print(f"  - Number of chunks: {len(X)}")
        print(f"  - Shape X: {X.shape}")
        print(f"  - Shape y: {y.shape}")
        
        return X, y, stats
    
    def get_validation_chunks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation chunks from the reserved validation series
        Must be called after preprocess_train()
        """
        if not hasattr(self, '_val_series'):
            raise ValueError("Must call preprocess_train() first")
        
        all_chunks_features = []
        all_chunks_segments = []
        
        for features, segments in self._val_series:
            chunks = self.create_chunks(features, segments)
            
            for chunk_features, chunk_segments in chunks:
                all_chunks_features.append(chunk_features)
                all_chunks_segments.append(chunk_segments)
        
        # Convertir a arrays numpy
        X_val = np.array(all_chunks_features)  # (n_chunks, chunk_length, 2)
        y_val = np.array(all_chunks_segments)  # (n_chunks, chunk_length)
        
        # Estandarizar usando el scaler del entrenamiento
        if self.config.normalize:
            X_val_reshaped = X_val.reshape(-1, 2)
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_val = X_val_scaled.reshape(X_val.shape).astype(np.float32)
        
        return X_val, y_val
    
    def preprocess_test(self, test_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """
        Preprocesar datos de test (series completas sin chunks)
        Returns: (X_series, y_series, stats)
        """
        # Cargar datos
        data = self.load_raw_data(test_path)
        
        # Estandarizar usando el scaler entrenado
        X_series = []
        y_series = []
        
        for features, segments in data:
            # Estandarizar
            if self.config.normalize:
                features_scaled = self.scaler.transform(features).astype(np.float32)
            else:
                features_scaled = features.astype(np.float32)
            
            X_series.append(features_scaled)
            y_series.append(segments)
        
        # Estadísticas
        y_all = np.concatenate(y_series)
        stats = self._compute_stats(y_all, "Test")
        
        print(f"\nDataset de test:")
        print(f"  - Number of series: {len(X_series)}")
        print(f"  - Lengths: min={min(len(x) for x in X_series)}, max={max(len(x) for x in X_series)}")
        
        return X_series, y_series, stats
    
    def _compute_stats(self, y: np.ndarray, dataset_name: str) -> Dict:
        """Calcular estadísticas de distribución de clases"""
        classes = list(self.segment_mapping.values())
        class_names = self.config.target_segments + [self.config.other_segment]
        
        # Flatten si es 2D (chunks)
        y_flat = y.flatten() if y.ndim > 1 else y
        
        # Contar
        counts = {}
        total = len(y_flat)
        
        for class_idx, class_name in enumerate(class_names):
            count = np.sum(y_flat == class_idx)
            percentage = (count / total) * 100 if total > 0 else 0
            counts[class_name] = {
                'count': int(count),
                'percentage': percentage
            }
        
        # Print stats
        print(f"\nClass distribution - {dataset_name}:")
        for class_name, stats in counts.items():
            print(f"  {class_name}: {stats['count']:>6} ({stats['percentage']:>5.1f}%)")
        
        return {
            'class_counts': counts,
            'total_samples': int(total),
            'num_classes': self.config.num_classes
        }


if __name__ == "__main__":
    # Test preprocessing
    config = Config(experiment_name="test")
    preprocessor = DataPreprocessor(config)
    
    # Test train
    X_train, y_train, stats_train = preprocessor.preprocess_train(config.train_path)
    
    # Test test
    X_test, y_test, stats_test = preprocessor.preprocess_test(config.test_path)
