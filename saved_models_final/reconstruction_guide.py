
# ============================================================
# KODE UNTUK RECONSTRUCT MODELS DI LOCAL
# Gunakan kode ini di app.py Anda
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

def build_mobilenet(img_size=224, num_classes=5):
    """Rebuild MobileNetV2 dengan arsitektur yang sama"""

    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'  # Download fresh ImageNet weights
    )
    base.trainable = False

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
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def build_efficientnet(img_size=224, num_classes=5):
    """Rebuild EfficientNetB0 dengan arsitektur yang sama"""

    base = EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'  # Download fresh ImageNet weights
    )
    base.trainable = False

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
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Cara load di Streamlit:
def load_model_safe(model_name, model_dir):
    """Load model dengan cara yang aman"""

    if model_name == 'cnn_base':
        # CNN Base bisa dimuat langsung
        try:
            model = tf.keras.models.load_model(
                f'{model_dir}/cnn_base.h5',
                compile=False,
                safe_mode=False
            )
            return model
        except:
            return None

    elif model_name == 'mobilenet':
        # Rebuild MobileNet dengan ImageNet weights
        model = build_mobilenet()

        # Load weights dari training (hanya top layers)
        weights_path = f'{model_dir}/mobilenet.weights.h5'
        try:
            # Load dengan skip_mismatch=True untuk hanya load layer yang cocok
            model.load_weights(weights_path, skip_mismatch=True, by_name=True)
            return model
        except:
            # Jika gagal, return model dengan ImageNet weights saja
            return model

    elif model_name == 'efficientnet':
        # Rebuild EfficientNet dengan ImageNet weights
        model = build_efficientnet()

        weights_path = f'{model_dir}/efficientnet.weights.h5'
        try:
            model.load_weights(weights_path, skip_mismatch=True, by_name=True)
            return model
        except:
            return model

    return None
