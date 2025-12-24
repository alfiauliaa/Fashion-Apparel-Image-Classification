"""
Fashion Apparel Image Classification - Streamlit Web App
MODERN UI VERSION - Enhanced Design
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
    page_title="Fashion Classifier Pro",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Custom CSS with Modern Design
st.markdown("""
<style>
    /* Main Containers */
    .main-container {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Typography */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons & Inputs */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Prediction Box */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom Classes */
    .class-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .confidence-bar {
        height: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL BUILDING FUNCTIONS (Tetap sama)
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
# WEIGHT LOADING FUNCTIONS (Tetap sama)
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
# MODEL LOADING (Tetap sama)
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
    
    return models, metadata, class_names, histories, comparison_df

# ============================================================
# IMAGE PREPROCESSING (Tetap sama)
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
# PREDICTION (Tetap sama)
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
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_confidence_bars(probabilities):
    """Plot confidence bars with modern design"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Modern gradient colors
    colors = []
    for p in probs:
        if p == max(probs):
            # Gold gradient for top prediction
            colors.append('#FFD700')
        else:
            # Blue gradient for others
            colors.append('#667eea')
    
    bars = ax.barh(classes, probs, color=colors, edgecolor='white', linewidth=2, height=0.6)
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_title('Prediction Confidence Distribution', fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax.set_xlim([0, 1])
    ax.set_facecolor('#f8f9fa')
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                va='center', fontweight='bold', fontsize=11,
                color='#2c3e50')
    
    plt.tight_layout()
    return fig

def plot_training_history(history, model_name):
    """Plot training history with modern design"""
    if history is None:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Training Performance', fontsize=18, fontweight='bold', y=1.02)
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training', marker='o', 
                 linewidth=3, markersize=8, color='#667eea', alpha=0.8)
    axes[0].plot(history['val_accuracy'], label='Validation', marker='s', 
                 linewidth=3, markersize=8, color='#764ba2', alpha=0.8)
    axes[0].set_title('Accuracy Curve', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training', marker='o', 
                 linewidth=3, markersize=8, color='#FF6B6B', alpha=0.8)
    axes[1].plot(history['val_loss'], label='Validation', marker='s', 
                 linewidth=3, markersize=8, color='#FFA726', alpha=0.8)
    axes[1].set_title('Loss Curve', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("🚀 Loading AI Models..."):
    models, metadata, class_names, histories, comparison_df = load_all_models()

if models is None or metadata is None:
    st.error("❌ Failed to load models!")
    st.stop()

available_models = [name for name, model in models.items() if model is not None]

if not available_models:
    st.error("❌ No models loaded successfully!")
    st.stop()

# ============================================================
# SIDEBAR - Modern Design
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0; font-size: 1.8rem;'>👔</h1>
        <h3 style='color: white; margin: 10px 0;'>Fashion AI</h3>
        <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>Deep Learning Classifier</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["🏠 Dashboard", "📊 Analytics", "📈 Training", "🎯 Models", "🔮 Predict"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    
    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div style='font-size: 1.2rem; color: #667eea;'>🤖</div>
            <div class='metric-value' style='font-size: 1.5rem;'>{len(available_models)}/3</div>
            <div class='metric-label' style='font-size: 0.8rem;'>Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div style='font-size: 1.2rem; color: #764ba2;'>📦</div>
            <div class='metric-value' style='font-size: 1.5rem;'>{len(class_names)}</div>
            <div class='metric-label' style='font-size: 0.8rem;'>Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info("""
    **Fashion AI Classifier**
    
    Powered by:
    - TensorFlow
    - Streamlit
    - Deep Learning
    
    UAP Machine Learning 2024
    """)

# ============================================================
# PAGES - Modern Design
# ============================================================

if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-title">Fashion AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced Deep Learning for Fashion Recognition</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📊 Overview</h2>', unsafe_allow_html=True)
    
    # Metrics Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total = metadata.get('total_train', 0) + metadata.get('total_test', 0) + metadata.get('total_val', 0)
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem; color: #667eea;'>📸</div>
            <div class='metric-value'>{total:,}</div>
            <div class='metric-label'>Total Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem; color: #2ecc71;'>🎯</div>
            <div class='metric-value'>{len(class_names)}</div>
            <div class='metric-label'>Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if comparison_df is not None:
            best = comparison_df['Test Accuracy'].str.extract(r'(\d+\.\d+)')[0].astype(float).max()
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2rem; color: #f39c12;'>🏆</div>
                <div class='metric-value'>{best:.1f}%</div>
                <div class='metric-label'>Best Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size: 2rem; color: #f39c12;'>🏆</div>
                <div class='metric-value'>N/A</div>
                <div class='metric-label'>Best Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem; color: #e74c3c;'>⚡</div>
            <div class='metric-value'>{len(available_models)}</div>
            <div class='metric-label'>Active Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Categories Section
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">👔 Fashion Categories</h2>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    icons = {'dress':'👗', 'pants':'👖', 'shirt':'👕', 'shoes':'👠', 'shorts':'🩳',
             'suit':'🤵', 'jacket':'🧥', 'hat':'🧢', 'bag':'👜', 'watch':'⌚'}
    
    for i, cls in enumerate(class_names):
        with cols[i % 5]:
            icon = icons.get(cls, '👔')
            st.markdown(f"""
            <div style='text-align:center; padding:1.5rem; background:#f8f9fa; border-radius:15px; border:2px solid #e0e0e0; transition:all 0.3s;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>{icon}</div>
                <h3 style='margin:0; color:#2c3e50; font-weight:600;'>{cls.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Comparison
    if comparison_df is not None:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">🏆 Model Performance</h2>', unsafe_allow_html=True)
        
        # Style the dataframe
        styled_df = comparison_df.style.background_gradient(subset=['Test Accuracy', 'Test Loss'], 
                                                          cmap='RdYlGn')
        st.dataframe(styled_df, use_container_width=True, height=200)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🚀 Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "📊 Analytics"
    
    with col2:
        if st.button("🔮 Make Prediction", use_container_width=True):
            st.session_state.page = "🔮 Predict"
    
    with col3:
        if st.button("📈 Training History", use_container_width=True):
            st.session_state.page = "📈 Training"
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📊 Analytics":
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title">📊 Dataset Analytics</h1>', unsafe_allow_html=True)
    
    # Dataset Statistics Cards
    col1, col2, col3 = st.columns(3)
    
    train_count = metadata.get('total_train', 0)
    test_count = metadata.get('total_test', 0)
    val_count = metadata.get('total_val', 0)
    total = train_count + test_count + val_count
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem; color: #3498db;'>📚</div>
            <div class='metric-value'>{train_count:,}</div>
            <div class='metric-label'>Training Data</div>
            <div style='color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;'>
                {train_count/total*100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem; color: #2ecc71;'>🧪</div>
            <div class='metric-value'>{test_count:,}</div>
            <div class='metric-label'>Test Data</div>
            <div style='color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;'>
                {test_count/total*100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem; color: #f39c12;'>✅</div>
            <div class='metric-value'>{val_count:,}</div>
            <div class='metric-label'>Validation Data</div>
            <div style='color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;'>
                {val_count/total*100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Split Visualization
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📈 Data Distribution</h2>', unsafe_allow_html=True)
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [train_count, test_count, val_count]
    labels = ['Training', 'Test', 'Validation']
    colors = ['#3498db', '#2ecc71', '#f39c12']
    explode = (0.1, 0, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.axis('equal')
    ax.set_title('Dataset Split Distribution', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Training":
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title">📈 Training History</h1>', unsafe_allow_html=True)
    
    # Model selector with tabs
    model_options = {
        'CNN Base': 'cnn_base',
        'MobileNetV2': 'mobilenet',
        'EfficientNetB0': 'efficientnet'
    }
    
    selected_display = st.selectbox("Select Model", list(model_options.keys()))
    selected_key = model_options[selected_display]
    
    st.markdown('</div>', unsafe_allow_html=True)