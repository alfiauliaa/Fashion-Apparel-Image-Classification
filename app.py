"""
Fashion Apparel Image Classification - Streamlit Web App
FIXED V3 - With Metadata and Training History Display
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fashion Apparel Classifier",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL BUILDING FUNCTIONS
# ============================================================

def build_cnn_base(img_size=224, num_classes=5):
    """Rebuild CNN Base architecture"""
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Conv2D(256, (3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_mobilenet(img_size=224, num_classes=5):
    """Rebuild MobileNetV2 with fresh ImageNet weights"""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_efficientnet(img_size=224, num_classes=5):
    """Rebuild EfficientNetB0 with fresh ImageNet weights"""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
    
    base = EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ============================================================
# WEIGHT LOADING FUNCTIONS
# ============================================================

def load_top_weights_from_npz(model, weights_path, info_path):
    """Load weights dari .npz file"""
    
    if not os.path.exists(weights_path):
        return False, "Weights file not found"
    
    try:
        weights_data = np.load(weights_path)
        
        with open(info_path, 'r') as f:
            weights_info = json.load(f)
        
        base_found = False
        layer_idx = 0
        
        for layer in model.layers:
            layer_name = layer.name
            
            if 'mobilenet' in layer_name.lower() or 'efficientnet' in layer_name.lower():
                base_found = True
                continue
            
            if base_found and len(layer.get_weights()) > 0:
                weights_to_load = []
                num_weights = len(layer.get_weights())
                
                found = False
                for stored_layer_name in weights_info['layer_info'].keys():
                    if layer_name in stored_layer_name or stored_layer_name in layer_name:
                        for i in range(num_weights):
                            key = f"{stored_layer_name}_w{i}"
                            if key in weights_data:
                                weights_to_load.append(weights_data[key])
                        
                        if len(weights_to_load) == num_weights:
                            found = True
                            break
                
                if not found:
                    stored_layers = list(weights_info['layer_info'].keys())
                    if layer_idx < len(stored_layers):
                        stored_layer_name = stored_layers[layer_idx]
                        for i in range(num_weights):
                            key = f"{stored_layer_name}_w{i}"
                            if key in weights_data:
                                weights_to_load.append(weights_data[key])
                
                if len(weights_to_load) == num_weights:
                    try:
                        current_weights = layer.get_weights()
                        shapes_match = all(
                            w.shape == cw.shape 
                            for w, cw in zip(weights_to_load, current_weights)
                        )
                        
                        if shapes_match:
                            layer.set_weights(weights_to_load)
                    except Exception as e:
                        pass
                
                layer_idx += 1
        
        return True, "Weights loaded successfully"
        
    except Exception as e:
        return False, f"Error loading weights: {str(e)}"

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_single_model(model_name, model_dir, img_size, num_classes):
    """Load model with proper weight loading"""
    
    tf.keras.backend.clear_session()
    
    if model_name == 'cnn_base':
        h5_path = model_dir / 'cnn_base.h5'
        if h5_path.exists():
            try:
                model = tf.keras.models.load_model(
                    str(h5_path),
                    compile=False,
                    safe_mode=False
                )
                return model, "Full model loaded"
            except Exception as e:
                return None, str(e)
    
    elif model_name == 'mobilenet':
        try:
            model = build_mobilenet(img_size, num_classes)
            
            weights_path = model_dir / 'mobilenet_top_weights.npz'
            info_path = model_dir / 'mobilenet_weights_info.json'
            
            if weights_path.exists() and info_path.exists():
                success, message = load_top_weights_from_npz(model, str(weights_path), str(info_path))
                
                if success:
                    return model, "Trained weights loaded from NPZ"
                else:
                    return model, f"ImageNet only: {message}"
            else:
                return model, "ImageNet weights only"
            
        except Exception as e:
            return None, str(e)
    
    elif model_name == 'efficientnet':
        try:
            model = build_efficientnet(img_size, num_classes)
            
            weights_path = model_dir / 'efficientnet_top_weights.npz'
            info_path = model_dir / 'efficientnet_weights_info.json'
            
            if weights_path.exists() and info_path.exists():
                success, message = load_top_weights_from_npz(model, str(weights_path), str(info_path))
                
                if success:
                    return model, "Trained weights loaded from NPZ"
                else:
                    return model, f"ImageNet only: {message}"
            else:
                return model, "ImageNet weights only"
            
        except Exception as e:
            return None, str(e)
    
    return None, "Unknown model"

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all models and metadata"""
    
    possible_dirs = ['saved_models_final', 'saved_models_fixed', 'saved_models2']
    model_dir = None
    
    for dirname in possible_dirs:
        path = Path(dirname)
        if path.exists():
            model_dir = path
            break
    
    if model_dir is None:
        st.error("❌ Model folder not found!")
        return None, None, None, None, None
    
    st.info(f"📁 Using folder: {model_dir}")
    
    # Load metadata
    try:
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        with open(model_dir / "class_indices.json", 'r') as f:
            class_indices = json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading metadata: {e}")
        return None, None, None, None, None
    
    img_size = metadata.get('img_size', 224)
    num_classes = metadata.get('num_classes', 5)
    
    # Load models
    st.info("🔄 Loading models...")
    models = {}
    load_status = {}
    
    model_configs = [
        ('cnn_base', 'CNN Base'),
        ('mobilenet', 'MobileNetV2'),
        ('efficientnet', 'EfficientNetB0')
    ]
    
    for model_name, display_name in model_configs:
        model, status = load_single_model(model_name, model_dir, img_size, num_classes)
        models[display_name] = model
        load_status[display_name] = status
    
    # Load histories
    histories = {}
    for model_name in ['cnn_base', 'mobilenet', 'efficientnet']:
        try:
            history_path = model_dir / f"{model_name}_history.pkl"
            if history_path.exists():
                with open(history_path, 'rb') as f:
                    histories[model_name] = pickle.load(f)
            else:
                histories[model_name] = None
        except:
            histories[model_name] = None
    
    # Load comparison
    try:
        comparison_path = model_dir / "model_comparison.csv"
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
        else:
            comparison_df = None
    except:
        comparison_df = None
    
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    
    st.markdown("### Load Status:")
    for model_name, status in load_status.items():
        if models[model_name] is not None:
            st.success(f"✅ {model_name}: {status}")
        else:
            st.error(f"❌ {model_name}: {status}")
    
    return models, metadata, class_names, histories, comparison_df

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image, model_name, img_size=224):
    """Preprocess image for each model"""
    
    image = image.resize((img_size, img_size))
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype(np.float32)
    
    if model_name == 'CNN Base':
        img_array = img_array / 255.0
    elif model_name == 'MobileNetV2':
        img_array = img_array / 127.5 - 1.0
    elif model_name == 'EfficientNetB0':
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        img_array = (img_array - mean) / std
    
    return np.expand_dims(img_array, axis=0)

# ============================================================
# PREDICTION
# ============================================================

def predict_image(image, model, model_name, class_names):
    """Make prediction"""
    
    if model is None:
        return None, None, None
    
    processed = preprocess_image(image, model_name)
    
    try:
        predictions = model.predict(processed, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])
        pred_class = class_names[pred_idx]
        
        all_probs = {class_names[i]: float(predictions[0][i]) 
                     for i in range(len(class_names))}
        
        return pred_class, confidence, all_probs
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# ============================================================
# VISUALIZATION
# ============================================================

def plot_confidence_bars(probabilities):
    """Plot confidence bars"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#2ecc71' if p == max(probs) else '#3498db' for p in probs]
    
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 0.01, i, f'{prob*100:.1f}%',
                va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_training_history(history, model_name):
    """Plot training history"""
    if history is None:
        st.warning(f"⚠️ No training history available for {model_name}")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================
# LOAD DATA
# ============================================================

models, metadata, class_names, histories, comparison_df = load_all_models()

if models is None or metadata is None:
    st.error("❌ Failed to load models!")
    st.stop()

available_models = [name for name, model in models.items() if model is not None]

if not available_models:
    st.error("❌ No models loaded successfully!")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", 
    ["🏠 Home", "📊 Dataset Overview", "📈 Training History", "🎯 Model Details", "🔮 Predict"])

st.sidebar.markdown("---")
st.sidebar.info(f"""
**System Info**
- TF: {tf.__version__}
- Models: {len(available_models)}/3
- Classes: {len(class_names)}
""")

# ============================================================
# PAGES
# ============================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">👔 Fashion Classifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Klasifikasi Jenis Pakaian Menggunakan Deep Learning</div>',
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Classes", len(class_names))
    with col2:
        st.metric("🤖 Models", f"{len(available_models)}/3")
    with col3:
        if comparison_df is not None:
            best = comparison_df['Test Accuracy'].str.extract(r'(\d+\.\d+)')[0].astype(float).max()
            st.metric("🏆 Best Accuracy", f"{best:.1f}%")
        else:
            st.metric("🏆 Best Accuracy", "N/A")
    with col4:
        total_images = metadata.get('total_train', 0) + metadata.get('total_test', 0) + metadata.get('total_val', 0)
        st.metric("📸 Total Images", f"{total_images:,}")
    
    st.markdown("---")
    
    # Classes
    st.subheader("📦 Fashion Categories")
    
    cols = st.columns(5)
    icons = {'dress':'👗', 'pants':'👖', 'shirt':'👕', 'shoes':'👞', 'shorts':'🩳'}
    
    for i, cls in enumerate(class_names):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center; padding:1.5rem; background:#f0f2f6; border-radius:0.5rem; border-left: 4px solid #1f77b4;'>
                <div style='font-size:3rem;'>{icons.get(cls, '👔')}</div>
                <b style='font-size:1.1rem;'>{cls.upper()}</b>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Comparison
    if comparison_df is not None:
        st.subheader("🏆 Model Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True, height=150)
    
    st.markdown("---")
    
    # Quick Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **3 Deep Learning Models**
          - CNN Base (Custom)
          - MobileNetV2 (Transfer Learning)
          - EfficientNetB0 (Transfer Learning)
        
        - **High Accuracy Classification**
        - **Real-time Prediction**
        - **Detailed Analytics**
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Info")
        if metadata:
            st.markdown(f"""
            - **Total Images**: {total_images:,}
            - **Training Set**: {metadata.get('total_train', 'N/A'):,}
            - **Test Set**: {metadata.get('total_test', 'N/A'):,}
            - **Validation Set**: {metadata.get('total_val', 'N/A'):,}
            - **Image Size**: {metadata.get('img_size', 224)}x{metadata.get('img_size', 224)}
            """)

elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    
    st.markdown("---")
    
    # Dataset Statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total = metadata.get('total_train', 0) + metadata.get('total_test', 0) + metadata.get('total_val', 0)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata.get('total_train', 0):,}</div>
            <div class="metric-label">Training Set (70%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata.get('total_test', 0):,}</div>
            <div class="metric-label">Test Set (20%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata.get('total_val', 0):,}</div>
            <div class="metric-label">Validation Set (10%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Split Ratio Visualization
    st.subheader("📊 Data Split Distribution")
    
    split_data = {
        'Split': ['Training', 'Test', 'Validation'],
        'Images': [
            metadata.get('total_train', 0),
            metadata.get('total_test', 0),
            metadata.get('total_val', 0)
        ],
        'Percentage': [70, 20, 10]
    }
    split_df = pd.DataFrame(split_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(split_df['Split'], split_df['Images'], 
                      color=['#3498db', '#2ecc71', '#f39c12'],
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, split_df['Images']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.dataframe(split_df, use_container_width=True, height=200)
    
    st.markdown("---")
    
    # Dataset Configuration
    st.subheader("⚙️ Dataset Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Image Properties:**
        - Image Size: {}x{}
        - Channels: RGB (3)
        - Format: JPEG/PNG
        - Batch Size: {}
        """.format(
            metadata.get('img_size', 224),
            metadata.get('img_size', 224),
            metadata.get('batch_size', 32)
        ))
    
    with col2:
        st.markdown("""
        **Preprocessing:**
        - Rescaling: 0-1 (CNN Base)
        - MobileNet preprocessing
        - EfficientNet preprocessing
        - No augmentation (original)
        """)

elif page == "📈 Training History":
    st.title("📈 Training History")
    
    st.markdown("---")
    
    # Model selector
    model_options = {
        'CNN Base': 'cnn_base',
        'MobileNetV2': 'mobilenet',
        'EfficientNetB0': 'efficientnet'
    }
    
    selected_display = st.selectbox("Select Model", list(model_options.keys()))
    selected_key = model_options[selected_display]
    
    st.markdown("---")
    
    # Show training curves
    if histories.get(selected_key) is not None:
        history = histories[selected_key]
        
        # Training summary
        st.subheader(f"📊 {selected_display} - Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            final_acc = history['accuracy'][-1] if 'accuracy' in history else 0
            st.metric("Final Train Acc", f"{final_acc*100:.2f}%")
        
        with col2:
            final_val_acc = history['val_accuracy'][-1] if 'val_accuracy' in history else 0
            st.metric("Final Val Acc", f"{final_val_acc*100:.2f}%")
        
        with col3:
            best_val_acc = max(history['val_accuracy']) if 'val_accuracy' in history else 0
            st.metric("Best Val Acc", f"{best_val_acc*100:.2f}%")
        
        with col4:
            num_epochs = len(history['accuracy']) if 'accuracy' in history else 0
            st.metric("Epochs Trained", num_epochs)
        
        st.markdown("---")
        
        # Plot training curves
        st.subheader("📈 Training Curves")
        fig = plot_training_history(history, selected_display)
        if fig:
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Detailed metrics table
        st.subheader("📋 Epoch-by-Epoch Metrics")
        
        epochs_data = {
            'Epoch': list(range(1, len(history['accuracy']) + 1)),
            'Train Accuracy': [f"{acc*100:.2f}%" for acc in history['accuracy']],
            'Val Accuracy': [f"{acc*100:.2f}%" for acc in history['val_accuracy']],
            'Train Loss': [f"{loss:.4f}" for loss in history['loss']],
            'Val Loss': [f"{loss:.4f}" for loss in history['val_loss']]
        }
        
        epochs_df = pd.DataFrame(epochs_data)
        st.dataframe(epochs_df, use_container_width=True, height=400)
        
        # Best epoch info
        best_epoch = np.argmax(history['val_accuracy']) + 1
        st.info(f"🏆 **Best Validation Accuracy** achieved at **Epoch {best_epoch}** with **{max(history['val_accuracy'])*100:.2f}%**")
    
    else:
        st.warning(f"⚠️ Training history not available for {selected_display}")
        st.info("Training history files (.pkl) might be missing from the model directory.")

elif page == "🎯 Model Details":
    st.title("🎯 Model Details")
    
    st.markdown("---")
    
    model_option = st.selectbox("Select Model", available_models)
    
    st.markdown("---")
    
    model_key_map = {
        'CNN Base': 'cnn_base',
        'MobileNetV2': 'mobilenet',
        'EfficientNetB0': 'efficientnet'
    }
    model_key = model_key_map.get(model_option)
    
    # Model Information
    if 'model_info' in metadata and model_key in metadata['model_info']:
        model_info = metadata['model_info'][model_key]
        
        st.subheader(f"📦 {model_option} Architecture")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{model_info.get('type', 'N/A').replace('_', ' ').title()}</div>
                <div class="metric-label">Model Type</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            params = model_info.get('total_params', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{params:,}</div>
                <div class="metric-label">Total Parameters</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            acc = model_info.get('test_accuracy', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            loss = model_info.get('test_loss', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{loss:.4f}</div>
                <div class="metric-label">Test Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model specifications
        st.subheader("🔧 Model Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Architecture Details:**")
            st.markdown(f"""
            - **Name**: {model_info.get('name', 'N/A')}
            - **Type**: {model_info.get('type', 'N/A').replace('_', ' ').title()}
            - **Base Model**: {model_info.get('base_model', 'Custom CNN')}
            - **Pretrained Weights**: {model_info.get('pretrained_weights', 'None')}
            - **Total Parameters**: {model_info.get('total_params', 0):,}
            """)
        
        with col2:
            st.markdown("**Training Configuration:**")
            st.markdown(f"""
            - **Input Shape**: {model_info.get('input_shape', [224, 224, 3])}
            - **Preprocessing**: {model_info.get('preprocessing', 'N/A')}
            - **Test Accuracy**: {model_info.get('test_accuracy', 0)*100:.2f}%
            - **Test Loss**: {model_info.get('test_loss', 0):.4f}
            """)
        
        st.markdown("---")
        
        # Model summary
        if models[model_option] is not None:
            st.subheader("📋 Model Architecture Summary")
            
            # Create summary string
            stringlist = []
            models[model_option].summary(print_fn=lambda x: stringlist.append(x))
            summary_string = "\n".join(stringlist)
            
            st.text(summary_string)
        
        st.markdown("---")
        
        # Training history for this model
        st.subheader("📈 Training Performance")
        
        if histories.get(model_key) is not None:
            history = histories[model_key]
            fig = plot_training_history(history, model_option)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("⚠️ Training history not available for this model")
    
    else:
        st.warning(f"⚠️ Detailed information not available for {model_option}")

elif page == "🔮 Predict":
    st.title("🔮 Fashion Prediction")
    
    st.markdown("---")
    
    # Model selection
    st.subheader("⚙️ Select Model")
    selected = st.selectbox("Choose a model for prediction", available_models, index=0)
    
    if models[selected] is None:
        st.error(f"❌ {selected} not available!")
        st.stop()
    
    # Show model info
    model_key_map = {
        'CNN Base': 'cnn_base',
        'MobileNetV2': 'mobilenet',
        'EfficientNetB0': 'efficientnet'
    }
    model_key = model_key_map.get(selected)
    
    if 'model_info' in metadata and model_key in metadata['model_info']:
        model_info = metadata['model_info'][model_key]
        acc = model_info.get('test_accuracy', 0)
        
        st.info(f"📊 **{selected}** - Test Accuracy: **{acc*100:.2f}%**")
    
    st.markdown("---")
    
    # Input method
    st.subheader("📥 Input Method")
    method = st.radio("Choose input method", ["📁 Upload Image", "📷 Camera"], horizontal=True)
    
    st.markdown("---")
    
    image = None
    
    if method == "📁 Upload Image":
        uploaded = st.file_uploader("Upload a fashion item image", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            image = Image.open(uploaded)
    else:
        camera = st.camera_input("Take a photo of the fashion item")
        if camera:
            image = Image.open(camera)
    
    if image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(image, use_container_width=True, caption="Original Image")
        
        with col2:
            st.subheader("🔄 Processing")
            
            with st.spinner("Analyzing image..."):
                pred_class, confidence, all_probs = predict_image(
                    image, models[selected], selected, class_names
                )
            
            if pred_class:
                # Prediction result
                icons = {'dress':'👗', 'pants':'👖', 'shirt':'👕', 'shoes':'👞', 'shorts':'🩳'}
                icon = icons.get(pred_class, '👔')
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style='margin:0; color:white;'>Prediction Result</h3>
                    <div style='font-size:4rem; margin:1rem 0;'>{icon}</div>
                    <h1 style='margin:0; color:white;'>{pred_class.upper()}</h1>
                    <h2 style='margin:0.5rem 0; color:white;'>{confidence*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence level
                if confidence >= 0.9:
                    st.success("🎯 **Very High Confidence** - Model is very certain about this prediction!")
                elif confidence >= 0.7:
                    st.info("✅ **High Confidence** - Model is confident about this prediction.")
                elif confidence >= 0.5:
                    st.warning("⚠️ **Moderate Confidence** - Model has some uncertainty.")
                else:
                    st.error("❌ **Low Confidence** - Model is uncertain about this prediction.")
        
        # Detailed probabilities
        if all_probs:
            st.markdown("---")
            st.subheader("📊 Detailed Prediction Probabilities")
            
            # Bar chart
            fig = plot_confidence_bars(all_probs)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Probability table
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Probability Table")
                prob_df = pd.DataFrame({
                    'Class': list(all_probs.keys()),
                    'Probability': [f"{v*100:.2f}%" for v in all_probs.values()],
                    'Confidence Score': list(all_probs.values())
                })
                prob_df = prob_df.sort_values('Confidence Score', ascending=False)
                prob_df = prob_df.drop('Confidence Score', axis=1)
                st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
            
            with col2:
                st.subheader("🎯 Top 3 Predictions")
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for i, (cls, prob) in enumerate(sorted_probs):
                    medal = ['🥇', '🥈', '🥉'][i]
                    st.markdown(f"""
                    <div style='padding:0.5rem; margin:0.5rem 0; background:#f0f2f6; border-radius:0.5rem;'>
                        {medal} <b>{cls.upper()}</b>: {prob*100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Please upload an image or take a photo to start prediction")
        
        st.markdown("---")
        st.subheader("💡 Tips for Best Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Image Quality:**
            - Use clear, well-lit images
            - Avoid blurry photos
            - Center the item in frame
            - Use plain backgrounds
            """)
        
        with col2:
            st.markdown("""
            **Supported Items:**
            - 👗 Dresses
            - 👖 Pants
            - 👕 Shirts
            - 👞 Shoes
            - 🩳 Shorts
            """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; padding:2rem; background:#f0f2f6; border-radius:0.5rem; margin-top:2rem;'>
    <h3 style='margin:0; color:#1f77b4;'>👔 Fashion Apparel Classifier</h3>
    <p style='margin:0.5rem 0;'><b>UAP Machine Learning 2024</b></p>
    <p style='margin:0; font-size:0.9rem;'>
        Deep Learning Image Classification | CNN Base · MobileNetV2 · EfficientNetB0
    </p>
</div>
""", unsafe_allow_html=True)