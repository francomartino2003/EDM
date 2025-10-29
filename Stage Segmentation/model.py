"""
Modelo TCN (Temporal Convolutional Network) para segmentación de stages
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    """Bloque de convolución dilatada"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        
        # Calcular padding para mantener longitud exacta
        # Para kernel impar: padding = (k-1)*d/2 mantiene longitud
        # Para kernel par: puede perder 1 timestep con ciertos dilation
        # Usamos ceiling para asegurar que no perdamos timesteps
        padding = ((kernel_size - 1) * dilation + 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation,
            padding=padding,
            padding_mode='replicate'  # Replicar bordes para mantener longitud
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCN(nn.Module):
    """
    Temporal Convolutional Network para segmentación temporal
    Input: (batch, 2, length) - [Voltage, Z]
    Output: (batch, num_classes, length) - probabilidades por clase
    """
    
    def __init__(self, config, debug=False):
        super().__init__()
        
        self.config = config
        self.debug = debug
        
        # Bloques de convolución dilatada
        self.conv_blocks = nn.ModuleList()
        
        in_channels = 2  # Input: Voltage y Z
        for out_channels, dilation in zip(config.channels, config.dilations):
            self.conv_blocks.append(
                DilatedConvBlock(in_channels, out_channels, config.kernel_size, dilation, config.dropout)
            )
            in_channels = out_channels
        
        # Capa final de clasificación
        self.classifier = nn.Conv1d(in_channels, config.num_classes, kernel_size=1)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos de las capas"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, 2, length)
        Returns:
            logits: (batch, num_classes, length)
        """
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Pasar por bloques convolucionales
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            if self.debug:
                print(f"After block {i}: {x.shape}")
        
        # Clasificación final
        logits = self.classifier(x)
        
        if self.debug:
            print(f"Final output shape: {logits.shape}\n")
        return logits
    
    def predict_proba(self, x):
        """Predecir probabilidades"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """Predecir clases"""
        probs = self.predict_proba(x)
        predictions = torch.argmax(probs, dim=1)
        return predictions
    
    def get_receptive_field(self):
        """Obtener receptive field del modelo"""
        return self.config.receptive_field


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar clases desbalanceadas
    """
    
    def __init__(self, alpha=None, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes, length)
            targets: (batch, length)
        """
        # Reshape para CrossEntropy
        logits_flat = logits.permute(0, 2, 1).contiguous().view(-1, logits.size(1))
        targets_flat = targets.view(-1)
        
        # Cross entropy
        ce_loss = F.cross_entropy(logits_flat, targets_flat, 
                                   weight=self.class_weights, 
                                   reduction='none')
        
        # Focal term
        probs = F.softmax(logits_flat, dim=1)
        p_t = probs.gather(1, targets_flat.view(-1, 1)).squeeze()
        
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets_flat)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def create_model(config, debug=False):
    """Factory function para crear modelo"""
    model = TCN(config, debug=debug)
    return model


def count_parameters(model):
    """Contar parámetros del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    from config import Config
    
    config = Config(experiment_name="test")
    model = create_model(config)
    
    print(f"Modelo creado:")
    print(f"  - Parámetros: {count_parameters(model):,}")
    print(f"  - Receptive field: {model.get_receptive_field()}")
    
    # Test forward
    batch_size = 4
    length = 512
    x = torch.randn(batch_size, 2, length)
    
    print(f"\nTest forward:")
    print(f"  Input shape: {x.shape}")
    out = model(x)
    print(f"  Output shape: {out.shape}")
    
    # Test predict
    predictions = model.predict(x)
    print(f"  Predictions shape: {predictions.shape}")
