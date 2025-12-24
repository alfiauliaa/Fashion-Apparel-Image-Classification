
# ============================================================
# HOW TO LOAD THESE MODELS
# ============================================================

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import numpy as np
import json

def load_model_with_trainable_weights(model_name, model_dir):
    """Load model dengan trainable weights"""
    
    if model_name == 'cnn_base':
        # Direct load
        model = tf.keras.models.load_model(
            f'{model_dir}/cnn_base.h5',
            compile=False,
            safe_mode=False
        )
        return model
    
    # Build base architecture
    if model_name == 'mobilenet':
        base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    elif model_name == 'efficientnet':
        base = EfficientNetB0(input_shape=(224,224,3), include_top=False, weights='imagenet')
    else:
        return None
    
    base.trainable = False
    
    # Build top layers - HARUS SAMA dengan saat training!
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(5, activation='softmax')
    ])
    
    # Load config
    config_path = f'{model_dir}/{model_name}_trainable_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load weights
    weights_path = f'{model_dir}/{model_name}_trainable.npz'
    weights_data = np.load(weights_path)
    
    # Set weights ke top layers (skip base model)
    top_layers = model.layers[1:]  # Skip base model
    
    for layer_idx, layer in enumerate(top_layers):
        if len(layer.get_weights()) > 0:
            layer_key = f"layer_{layer_idx}_{layer.__class__.__name__}"
            
            # Load weights untuk layer ini
            weights_to_set = []
            w_idx = 0
            while True:
                npz_key = f"{layer_key}_w{w_idx}"
                if npz_key in weights_data:
                    weights_to_set.append(weights_data[npz_key])
                    w_idx += 1
                else:
                    break
            
            # Set weights jika jumlah cocok
            if len(weights_to_set) == len(layer.get_weights()):
                layer.set_weights(weights_to_set)
    
    return model

# Usage:
# model_mobile = load_model_with_trainable_weights('mobilenet', 'saved_models_final')
# model_eff = load_model_with_trainable_weights('efficientnet', 'saved_models_final')
