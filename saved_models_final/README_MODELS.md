# Fashion Apparel Classification - Saved Models (Keras 3)

**IMPORTANT: These models were saved with Keras 3 (TensorFlow 2.19+)**

## Environment Info
- TensorFlow Version: 2.19.0
- Keras Version: 3.10.0
- Date: 2025-12-24 19:11:56

## Model Files

### Available Formats

Each model is saved in **2 formats**:

1. **`.keras` format** (RECOMMENDED - Native Keras 3)
   - Best compatibility with TensorFlow 2.16+
   - Preserves all model information
   
2. **`.h5` format** (Legacy)
   - Backward compatibility
   - May have some limitations in Keras 3

### Model List

#### 1. CNN Base (Non-Pretrained)
- Files: `cnn_base.keras`, `cnn_base.h5`
- Weights: `cnn_base.weights.h5`
- Accuracy: 93.31%
- Parameters: 19,400,901
- Preprocessing: Simple rescaling (0-1)

#### 2. MobileNetV2 (Transfer Learning)
- Files: `mobilenet.keras`, `mobilenet.h5`
- Weights: `mobilenet.weights.h5`
- Accuracy: 97.58%
- Parameters: 2,625,605
- Preprocessing: MobileNetV2 preprocessing

#### 3. EfficientNetB0 (Transfer Learning)
- Files: `efficientnet.keras`, `efficientnet.h5`
- Weights: `efficientnet.weights.h5`
- Accuracy: 98.16%
- Parameters: 4,417,192
- Preprocessing: EfficientNet preprocessing

## Loading Models (Keras 3)

### Prerequisites
```bash
# Make sure you have TensorFlow 2.16+ (Keras 3)
pip install tensorflow>=2.16.0

# Or use TensorFlow 2.19+
pip install tensorflow
```

### Method 1: Load .keras File (RECOMMENDED)
```python
import tensorflow as tf

# Load complete model
model = tf.keras.models.load_model('cnn_base.keras', compile=False)

# Ready to use
predictions = model.predict(image_array)
```

### Method 2: Load .h5 File
```python
# Load .h5 with safe_mode=False for Keras 3
model = tf.keras.models.load_model(
    'cnn_base.h5',
    compile=False,
    safe_mode=False
)
```

### Method 3: Load Weights Only
```python
# First rebuild the architecture, then:
model.load_weights('cnn_base.weights.h5')
```

## Preprocessing

```python
import numpy as np
from PIL import Image

# Load and resize image
image = Image.open('test.jpg').resize((224, 224))
img_array = np.array(image).astype(np.float32)

# Convert to RGB if needed
if len(img_array.shape) == 2:
    img_array = np.stack([img_array] * 3, axis=-1)
elif img_array.shape[2] == 4:
    img_array = img_array[:, :, :3]

# Preprocessing based on model
if model_name == 'CNN Base':
    img_array = img_array / 255.0
elif model_name == 'MobileNetV2':
    img_array = img_array / 127.5 - 1.0
elif model_name == 'EfficientNetB0':
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    img_array = (img_array - mean) / std

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
```

## Classes

dress, pants, shirt, shoes, shorts

## Training Details

- Image Size: 224x224
- Batch Size: 32
- Dataset Split: 70% train, 20% test, 10% validation
- Total Train: 4193
- Total Test: 1196
- Total Val: 604

## Troubleshooting

### If models fail to load:

**Problem: "Cannot convert to shape" or "BatchNormalization expects 1 input"**

This means you're trying to load Keras 3 models with Keras 2 (TF < 2.16).

**Solution:**
```bash
# Upgrade TensorFlow
pip install --upgrade tensorflow

# Or specifically
pip install tensorflow>=2.19.0
```

**Problem: "Invalid filepath extension"**

This is expected for SavedModel format in Keras 3. Use `.keras` or `.h5` instead.

### Version Compatibility

| Local TensorFlow | Status | Action |
|------------------|--------|--------|
| TF < 2.16 (Keras 2) | ❌ Incompatible | Upgrade TensorFlow |
| TF 2.16-2.18 (Keras 3) | ⚠️ Partial | Use `.keras` files |
| TF 2.19+ (Keras 3) | ✅ Fully Compatible | Use any format |

## Using with Streamlit

```python
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_base.keras', compile=False)

model = load_model()
```

## Additional Files

- `metadata.json` - Model configurations
- `class_indices.json` - Class mappings
- `model_comparison.csv` - Performance comparison
- `*_history.pkl` - Training histories
- `sample_images/` - Sample images per class

## Notes

- All models are saved with Keras 3 format
- `.keras` files are recommended over `.h5`
- Weights files use `.weights.h5` extension (Keras 3 requirement)
- Models were trained with data augmentation
- Use `compile=False` when loading to avoid optimizer warnings

## Generated: 2025-12-24 19:11:56
