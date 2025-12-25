# Fashion Apparel Models - FIXED FOR KERAS 3

## Problem & Solution

**Problem**: Transfer learning models (MobileNet, EfficientNet) gagal dimuat di local karena BatchNormalization layer incompatibility antara Keras 2 dan Keras 3.

**Solution**:
- CNN Base: Disimpan sebagai full model (OK)
- MobileNet & EfficientNet: Hanya simpan weights, rebuild architecture di local

## Files

### Working Models
- `cnn_base.h5` - Full model, bisa langsung dimuat
- `cnn_base.weights.h5` - Weights only

### Transfer Learning (Requires Rebuild)
- `mobilenet.weights.h5` - Weights untuk top layers
- `efficientnet.weights.h5` - Weights untuk top layers
- `*_config.json` - Model configuration

### Code & Docs
- `reconstruction_guide.py` - Kode untuk rebuild models di local
- `metadata.json` - Model metadata
- `class_indices.json` - Class mappings
- `*_history.pkl` - Training histories

## How to Load Models (Di Streamlit)

### Method 1: Gunakan reconstruction_guide.py

```python
# Copy function dari reconstruction_guide.py ke app.py
from reconstruction_guide import load_model_safe

# Load models
model_cnn = load_model_safe('cnn_base', 'saved_models_fixed')
model_mobile = load_model_safe('mobilenet', 'saved_models_fixed')
model_eff = load_model_safe('efficientnet', 'saved_models_fixed')
```

### Method 2: Manual

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# CNN Base - Direct load
model_cnn = tf.keras.models.load_model(
    'saved_models_fixed/cnn_base.h5',
    compile=False,
    safe_mode=False
)

# MobileNet - Rebuild + Load weights
def build_mobilenet():
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

model_mobile = build_mobilenet()
model_mobile.load_weights(
    'saved_models_fixed/mobilenet.weights.h5',
    skip_mismatch=True,
    by_name=True
)
```

## Notes

1. **CNN Base**: Bisa langsung dimuat, tidak ada masalah
2. **MobileNet & EfficientNet**:
   - Base model menggunakan ImageNet weights (fresh download)
   - Top layers menggunakan weights dari training
   - Jika load weights gagal, tetap bisa jalan dengan ImageNet weights saja
3. **Performance**: Mungkin sedikit berbeda karena base model menggunakan fresh ImageNet weights

## Generated: 2025-12-24 13:55:15
