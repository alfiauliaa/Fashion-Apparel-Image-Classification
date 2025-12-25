"""
Fashion Apparel Image Classification - Streamlit Web App
Dark Mode Design with Sample Dataset Images
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import pickle
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
import random
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

icons = {'dress':'👗', 'pants':'👖', 'shirt':'👕', 'shoes':'👞', 'shorts':'🩳'}

# ============================================================
# CONFIGURASI UKURAN GAMBAR SAMPLE
# ============================================================
SAMPLE_IMAGE_SIZE = (250, 250)  # Ukuran konsisten untuk semua gambar sample
SAMPLE_IMAGES_PER_CLASS = 3     # Jumlah gambar per kelas

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Dark Mode Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Dark Theme */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: #0f1419;
    }
    
    .main {
        background: #0f1419;
    }
    
    .block-container {
        background: #1a1f2e;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        border: 1px solid #2d3748;
    }
    
    /* Text Colors */
    .stMarkdown, p, span, div, label {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f7fafc !important;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #a0aec0;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadeInUp 0.8s ease;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: scaleIn 0.5s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .prediction-box * {
        color: white !important;
    }
    
    /* Metric Card - Dark */
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #4a5568;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0 !important;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Class Card - Dark */
    .class-card {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 15px;
        border: 2px solid #4a5568;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease;
    }
    
    .class-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .class-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .class-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Info Box - Dark */
    .info-box {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #4a5568;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .info-box h3, .info-box h4 {
        color: #667eea !important;
        margin-bottom: 1rem;
    }
    
    /* Section Header */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sample Image Grid */
    .sample-image-grid {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .sample-image-item {
        text-align: center;
        margin: 10px;
        transition: transform 0.3s ease;
    }
    
    .sample-image-item:hover {
        transform: translateY(-5px);
    }
    
    .sample-image-frame {
        border: 3px solid #4a5568;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 10px;
        background: #2d3748;
        padding: 5px;
        transition: border-color 0.3s ease;
    }
    
    .sample-image-frame:hover {
        border-color: #667eea;
    }
    
    .sample-image-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Sidebar Dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 1px solid #2d3748;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #667eea !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] p {
        color: #a0aec0 !important;
    }
    
    /* Buttons Dark */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #4a5568, transparent);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Dataframe Dark */
    .dataframe {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    
    /* Input fields dark */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* File uploader dark */
    [data-testid="stFileUploadDropzone"] {
        background: #2d3748 !important;
        border: 2px dashed #4a5568 !important;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: #2d3748 !important;
        border-left: 4px solid #667eea !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resize_image_with_padding(image, target_size):
    """
    Resize image dengan padding untuk menjaga aspect ratio
    dan menghasilkan ukuran yang konsisten
    """
    width, height = image.size
    target_width, target_height = target_size
    
    # Hitung rasio untuk resize
    width_ratio = target_width / width
    height_ratio = target_height / height
    ratio = min(width_ratio, height_ratio)
    
    # Resize dengan menjaga aspect ratio
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Tambahkan padding jika diperlukan
    if new_width < target_width or new_height < target_height:
        padded_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded_image.paste(resized_image, (paste_x, paste_y))
        return padded_image
    
    return resized_image

def load_sample_images(sample_dir='sample_images', num_per_class=SAMPLE_IMAGES_PER_CLASS):
    """Load sample images dari folder sample_images"""
    sample_images = {}
    
    if not os.path.exists(sample_dir):
        st.warning(f"📁 Folder '{sample_dir}' tidak ditemukan. Sample images tidak akan ditampilkan.")
        return None
    
    # Cari semua subdirektori (setiap kelas)
    for class_name in os.listdir(sample_dir):
        class_path = os.path.join(sample_dir, class_name)
        if os.path.isdir(class_path):
            images = []
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            if not image_files:
                continue
            
            # Pilih gambar secara random
            num_to_select = min(num_per_class, len(image_files))
            selected = random.sample(image_files, num_to_select)
            
            for img_file in selected:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path)
                    # Resize gambar ke ukuran yang konsisten
                    img_resized = resize_image_with_padding(img, SAMPLE_IMAGE_SIZE)
                    images.append(img_resized)
                except Exception as e:
                    st.warning(f"⚠️ Tidak bisa memuat gambar {img_file}: {str(e)}")
                    continue
            
            if images:
                sample_images[class_name] = images
    
    if not sample_images:
        st.warning(f"📁 Tidak ada gambar yang ditemukan di folder '{sample_dir}'")
        return None
    
    return sample_images

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
    """Plot confidence bars with dark theme"""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1f2e')
    ax.set_facecolor('#2d3748')
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = []
    for p in probs:
        if p == max(probs):
            colors.append('#667eea')
        else:
            colors.append('#4a5568')
    
    bars = ax.barh(classes, probs, color=colors, edgecolor='none', height=0.6)
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='600', color='#e2e8f0')
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='700', 
                 color='#667eea', pad=20)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.2, linestyle='--', color='#4a5568')
    ax.tick_params(colors='#e2e8f0')
    
    for spine in ax.spines.values():
        spine.set_color('#4a5568')
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 0.02, i, f'{prob*100:.1f}%',
                va='center', fontweight='600', fontsize=10, color='#e2e8f0')
    
    plt.tight_layout()
    return fig

def plot_training_history(history, model_name):
    """Plot training history with dark theme"""
    if history is None:
        st.warning(f"⚠️ No training history available for {model_name}")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1f2e')
    
    for ax in axes:
        ax.set_facecolor('#2d3748')
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training', marker='o', linewidth=2.5, 
                color='#667eea', markersize=6)
    axes[0].plot(history['val_accuracy'], label='Validation', marker='s', linewidth=2.5,
                color='#764ba2', markersize=6)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=13, fontweight='700', 
                     color='#e2e8f0', pad=15)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='600', color='#a0aec0')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='600', color='#a0aec0')
    axes[0].legend(frameon=True, fancybox=True, shadow=True, fontsize=10, 
                  facecolor='#2d3748', edgecolor='#4a5568', labelcolor='#e2e8f0')
    axes[0].grid(True, alpha=0.2, linestyle='--', color='#4a5568')
    axes[0].tick_params(colors='#a0aec0')
    for spine in axes[0].spines.values():
        spine.set_color('#4a5568')
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training', marker='o', linewidth=2.5,
                color='#667eea', markersize=6)
    axes[1].plot(history['val_loss'], label='Validation', marker='s', linewidth=2.5,
                color='#764ba2', markersize=6)
    axes[1].set_title(f'{model_name} - Loss', fontsize=13, fontweight='700', 
                     color='#e2e8f0', pad=15)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='600', color='#a0aec0')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='600', color='#a0aec0')
    axes[1].legend(frameon=True, fancybox=True, shadow=True, fontsize=10,
                  facecolor='#2d3748', edgecolor='#4a5568', labelcolor='#e2e8f0')
    axes[1].grid(True, alpha=0.2, linestyle='--', color='#4a5568')
    axes[1].tick_params(colors='#a0aec0')
    for spine in axes[1].spines.values():
        spine.set_color('#4a5568')
    
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

# Load sample images
sample_images = load_sample_images()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", 
    ["🏠 Home", "📊 Dataset Overview", "📈 Training History", "🎯 Model Details", "🔮 Predict"],
    label_visibility="collapsed")

st.sidebar.markdown("---")

# Settings for sample images
st.sidebar.markdown("### ⚙️ Sample Images Settings")
if st.sidebar.button("🔄 Reload Sample Images"):
    sample_images = load_sample_images()
    st.rerun()

st.sidebar.markdown(f"""
<div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; margin-top: 1rem; border: 1px solid #4a5568;'>
    <h4 style='margin:0; color: #667eea; font-weight: 700;'>⚡ System Info</h4>
    <p style='margin: 1rem 0 0 0; font-size: 0.9rem; line-height: 1.8; color: #a0aec0;'>
        <b style='color: #e2e8f0;'>TensorFlow:</b> {tf.__version__}<br>
        <b style='color: #e2e8f0;'>Models:</b> {len(available_models)}/3<br>
        <b style='color: #e2e8f0;'>Classes:</b> {len(class_names)}<br>
        <b style='color: #e2e8f0;'>Sample Size:</b> {SAMPLE_IMAGE_SIZE[0]}×{SAMPLE_IMAGE_SIZE[1]} px
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGES
# ============================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">👔 Fashion Classifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Fashion Item Classification System</div>',
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(class_names)}</div>
            <div class="metric-label">Fashion Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(available_models)}/3</div>
            <div class="metric-label">AI Models Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if comparison_df is not None:
            best = comparison_df['Test Accuracy'].str.extract(r'(\d+\.\d+)')[0].astype(float).max()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best:.1f}%</div>
                <div class="metric-label">Best Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">N/A</div>
                <div class="metric-label">Best Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        total_images = metadata.get('total_train', 0) + metadata.get('total_test', 0) + metadata.get('total_val', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_images:,}</div>
            <div class="metric-label">Training Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Classes with icons
    st.markdown('<div class="section-header">👗 Fashion Categories</div>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    icons = {'dress':'👗', 'pants':'👖', 'shirt':'👕', 'shoes':'👞', 'shorts':'🩳'}
    
    for i, cls in enumerate(class_names):
        with cols[i]:
            st.markdown(f"""
            <div class='class-card'>
                <div class='class-icon'>{icons.get(cls, '👔')}</div>
                <div class='class-name'>{cls}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Comparison
    if comparison_df is not None:
        st.markdown('<div class="section-header">🏆 Model Performance</div>', unsafe_allow_html=True)
        st.dataframe(comparison_df, use_container_width=True, height=150)
    
    st.markdown("---")
    
    # Quick Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3 style='margin:0 0 1rem 0; color: #667eea;'>🎯 Key Features</h3>
            <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                ✨ <b>3 Advanced AI Models</b><br>
                • Custom CNN Architecture<br>
                • MobileNetV2 (Transfer Learning)<br>
                • EfficientNetB0 (Transfer Learning)<br><br>
                🚀 <b>Capabilities</b><br>
                • High Accuracy Classification<br>
                • Real-time Predictions<br>
                • Detailed Performance Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3 style='margin:0 0 1rem 0; color: #667eea;'>📊 Dataset Statistics</h3>
            <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                📦 <b>Total Images:</b> {total_images:,}<br>
                🎓 <b>Training Set:</b> {metadata.get('total_train', 'N/A'):,} (70%)<br>
                🧪 <b>Test Set:</b> {metadata.get('total_test', 'N/A'):,} (20%)<br>
                ✅ <b>Validation Set:</b> {metadata.get('total_val', 'N/A'):,} (10%)<br>
                📐 <b>Image Size:</b> {metadata.get('img_size', 224)}×{metadata.get('img_size', 224)} pixels
            </p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Dataset Overview":
    st.markdown('<div class="main-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Statistics
    st.markdown('<div class="section-header">📈 Dataset Distribution</div>', unsafe_allow_html=True)
    
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
    
    # Sample Images Section
    if sample_images:
        st.markdown(f'<div class="section-header">📸 Sample Dataset Images ({SAMPLE_IMAGE_SIZE[0]}×{SAMPLE_IMAGE_SIZE[1]} pixels)</div>', unsafe_allow_html=True)
        
        for class_name in class_names:
            if class_name in sample_images:
                st.markdown(f"""
                <div class="info-box" style='margin: 1.5rem 0;'>
                    <h4 style='margin:0; color: #667eea; text-transform: uppercase; letter-spacing: 2px;'>
                        {icons.get(class_name, '👔')} {class_name} 
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Create columns for sample images
                num_images = len(sample_images[class_name])
                cols = st.columns(num_images)
                
                for idx, img in enumerate(sample_images[class_name]):
                    with cols[idx]:
                        # Display image dengan styling konsisten
                        st.markdown("""
                        <div style='text-align: center;'>
                            <div style='border: 2px solid #4a5568; border-radius: 10px; overflow: hidden; 
                                     margin-bottom: 10px; background: #2d3748; padding: 5px;'>
                        """, unsafe_allow_html=True)
                        
                        st.image(img, use_container_width=True, 
                                caption=f"Sample", 
                                output_format="PNG")
                        
                        st.markdown("""
                            </div>
           
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
    else:
        st.info("💡 Sample images will be displayed here if available in 'sample_images' folder")
        st.markdown("---")
    
    # Split Ratio Visualization
    st.markdown('<div class="section-header">📊 Data Split Analysis</div>', unsafe_allow_html=True)
    
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
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1f2e')
        ax.set_facecolor('#2d3748')
        bars = ax.bar(split_df['Split'], split_df['Images'], 
                      color=['#667eea', '#764ba2', '#f39c12'],
                      edgecolor='none', width=0.6)
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='600', color='#e2e8f0')
        ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='700', 
                    color='#667eea', pad=20)
        ax.grid(axis='y', alpha=0.2, linestyle='--', color='#4a5568')
        ax.tick_params(colors='#a0aec0')
        for spine in ax.spines.values():
            spine.set_color('#4a5568')
        
        for bar, count in zip(bars, split_df['Images']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontweight='600', fontsize=10, color='#e2e8f0')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.dataframe(split_df, use_container_width=True, height=200)
    
    st.markdown("---")
    
    # Dataset Configuration
    st.markdown('<div class="section-header">⚙️ Configuration Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='margin:0 0 1rem 0; color: #667eea;'>🖼️ Image Properties</h4>
            <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                • <b>Dimensions:</b> {metadata.get('img_size', 224)}×{metadata.get('img_size', 224)} pixels<br>
                • <b>Sample Display:</b> {SAMPLE_IMAGE_SIZE[0]}×{SAMPLE_IMAGE_SIZE[1]} pixels<br>
                • <b>Color Channels:</b> RGB (3 channels)<br>
                • <b>Format:</b> JPEG/PNG<br>
                • <b>Batch Size:</b> {metadata.get('batch_size', 32)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style='margin:0 0 1rem 0; color: #667eea;'>🔧 Preprocessing Pipeline</h4>
            <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                • <b>CNN Base:</b> Rescaling (0-1 normalization)<br>
                • <b>MobileNet:</b> Specialized preprocessing<br>
                • <b>EfficientNet:</b> Advanced preprocessing<br>
                • <b>Augmentation:</b> Original images only
            </p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📈 Training History":
    st.markdown('<div class="main-header">📈 Training History</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model selector
    model_options = {
        'CNN Base': 'cnn_base',
        'MobileNetV2': 'mobilenet',
        'EfficientNetB0': 'efficientnet'
    }
    
    selected_display = st.selectbox("Select Model to Analyze", list(model_options.keys()))
    selected_key = model_options[selected_display]
    
    st.markdown("---")
    
    # Show training curves
    if histories.get(selected_key) is not None:
        history = histories[selected_key]
        
        # Training summary
        st.markdown(f'<div class="section-header">📊 {selected_display} Performance</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            final_acc = history['accuracy'][-1] if 'accuracy' in history else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{final_acc*100:.2f}%</div>
                <div class="metric-label">Final Train Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            final_val_acc = history['val_accuracy'][-1] if 'val_accuracy' in history else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{final_val_acc*100:.2f}%</div>
                <div class="metric-label">Final Val Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_val_acc = max(history['val_accuracy']) if 'val_accuracy' in history else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_val_acc*100:.2f}%</div>
                <div class="metric-label">Best Val Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            num_epochs = len(history['accuracy']) if 'accuracy' in history else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_epochs}</div>
                <div class="metric-label">Epochs Trained</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Plot training curves
        st.markdown('<div class="section-header">📈 Learning Curves</div>', unsafe_allow_html=True)
        fig = plot_training_history(history, selected_display)
        if fig:
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Detailed metrics table
        st.markdown('<div class="section-header">📋 Detailed Metrics</div>', unsafe_allow_html=True)
        
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
        st.markdown(f"""
        <div class="info-box">
            <h4 style='margin:0; color: #667eea;'>🏆 Peak Performance</h4>
            <p style='margin:0.5rem 0 0 0; font-size: 1.1rem; color: #e2e8f0;'>
                Best validation accuracy of <b style='color: #667eea;'>{max(history['val_accuracy'])*100:.2f}%</b> achieved at <b style='color: #667eea;'>Epoch {best_epoch}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.warning(f"⚠️ Training history not available for {selected_display}")
        st.info("Training history files (.pkl) might be missing from the model directory.")

elif page == "🎯 Model Details":
    st.markdown('<div class="main-header">🎯 Model Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    model_option = st.selectbox("Select Model to Explore", available_models)
    
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
        
        st.markdown(f'<div class="section-header">📦 {model_option} Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{model_info.get('type', 'N/A').replace('_', ' ').title()}</div>
                <div class="metric-label">Architecture Type</div>
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
        st.markdown('<div class="section-header">🔧 Technical Specifications</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4 style='margin:0 0 1rem 0; color: #667eea;'>🏗️ Architecture Details</h4>
                <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                    • <b>Model Name:</b> {model_info.get('name', 'N/A')}<br>
                    • <b>Type:</b> {model_info.get('type', 'N/A').replace('_', ' ').title()}<br>
                    • <b>Base Model:</b> {model_info.get('base_model', 'Custom CNN')}<br>
                    • <b>Pretrained:</b> {model_info.get('pretrained_weights', 'None')}<br>
                    • <b>Parameters:</b> {model_info.get('total_params', 0):,}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4 style='margin:0 0 1rem 0; color: #667eea;'>⚙️ Training Configuration</h4>
                <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                    • <b>Input Shape:</b> {model_info.get('input_shape', [224, 224, 3])}<br>
                    • <b>Preprocessing:</b> {model_info.get('preprocessing', 'N/A')}<br>
                    • <b>Test Accuracy:</b> {model_info.get('test_accuracy', 0)*100:.2f}%<br>
                    • <b>Test Loss:</b> {model_info.get('test_loss', 0):.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model summary
        if models[model_option] is not None:
            st.markdown('<div class="section-header">📋 Layer Architecture</div>', unsafe_allow_html=True)
            
            # Create summary string
            stringlist = []
            models[model_option].summary(print_fn=lambda x: stringlist.append(x))
            summary_string = "\n".join(stringlist)
            
            st.text(summary_string)
        
        st.markdown("---")
        
        # Training history for this model
        st.markdown('<div class="section-header">📈 Training Performance</div>', unsafe_allow_html=True)
        
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
    st.markdown('<div class="main-header">🔮 Fashion Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model selection
    st.markdown('<div class="section-header">⚙️ Model Selection</div>', unsafe_allow_html=True)
    selected = st.selectbox("Choose your AI model", available_models, index=0)
    
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
        
        st.markdown(f"""
        <div class="info-box">
            <p style='margin:0; font-size: 1.1rem; color: #e2e8f0;'>
                🤖 <b style='color: #667eea;'>{selected}</b> | Test Accuracy: <b style='color: #667eea;'>{acc*100:.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input method
    st.markdown('<div class="section-header">📥 Upload Your Image</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-header">📸 Input Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="Original Image")
        
        with col2:
            st.markdown('<div class="section-header">🔄 AI Analysis</div>', unsafe_allow_html=True)
            
            with st.spinner("🔍 Analyzing image with AI..."):
                pred_class, confidence, all_probs = predict_image(
                    image, models[selected], selected, class_names
                )
            
            if pred_class:
                # Prediction result
                icons = {'dress':'👗', 'pants':'👖', 'shirt':'👕', 'shoes':'👞', 'shorts':'🩳'}
                icon = icons.get(pred_class, '👔')
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style='margin:0; font-weight: 400;'>Prediction Result</h3>
                    <div style='font-size:5rem; margin:1rem 0;'>{icon}</div>
                    <h1 style='margin:0; font-weight: 800; font-size: 2.5rem;'>{pred_class.upper()}</h1>
                    <h2 style='margin:0.5rem 0; font-weight: 700;'>{confidence*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence level
                if confidence >= 0.9:
                    st.success("🎯 **Very High Confidence** - The model is extremely certain!")
                elif confidence >= 0.7:
                    st.info("✅ **High Confidence** - Strong prediction detected")
                elif confidence >= 0.5:
                    st.warning("⚠️ **Moderate Confidence** - Some uncertainty present")
                else:
                    st.error("❌ **Low Confidence** - Consider using a clearer image")
        
        # Detailed probabilities
        if all_probs:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Bar chart
            fig = plot_confidence_bars(all_probs)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Probability table
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="section-header">📋 All Predictions</div>', unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    'Class': list(all_probs.keys()),
                    'Probability': [f"{v*100:.2f}%" for v in all_probs.values()],
                    'Confidence Score': list(all_probs.values())
                })
                prob_df = prob_df.sort_values('Confidence Score', ascending=False)
                prob_df = prob_df.drop('Confidence Score', axis=1)
                st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-header">🎯 Top 3</div>', unsafe_allow_html=True)
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for i, (cls, prob) in enumerate(sorted_probs):
                    medal = ['🥇', '🥈', '🥉'][i]
                    st.markdown(f"""
                    <div class="metric-card" style='margin: 0.5rem 0;'>
                        <div style='font-size: 1.5rem;'>{medal}</div>
                        <div style='font-weight: 700; color: #e2e8f0; margin: 0.3rem 0; text-transform: uppercase;'>{cls}</div>
                        <div style='color: #667eea; font-weight: 700; font-size: 1.2rem;'>{prob*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="info-box" style='text-align: center;'>
            <h3 style='margin:0; color: #667eea;'>👆 Ready to Classify!</h3>
            <p style='margin:0.5rem 0 0 0; color: #e2e8f0;'>Upload an image or take a photo to start the AI-powered fashion classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="section-header">💡 Tips for Best Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4 style='margin:0 0 1rem 0; color: #667eea;'>📸 Image Quality</h4>
                <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                    • Use clear, well-lit images<br>
                    • Avoid blurry or dark photos<br>
                    • Center the item in frame<br>
                    • Use plain backgrounds
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4 style='margin:0 0 1rem 0; color: #667eea;'>👔 Supported Items</h4>
                <p style='margin:0; line-height: 1.8; color: #e2e8f0;'>
                    👗 Dresses<br>
                    👖 Pants<br>
                    👕 Shirts<br>
                    👞 Shoes<br>
                    🩳 Shorts
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:20px; margin-top:3rem; box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4); border: 1px solid rgba(255,255,255,0.1);'>
    <h2 style='margin:0; color:white; font-weight: 800; letter-spacing: -1px;'>👔 Fashion Apparel Classifier</h2>
    <p style='margin:1rem 0 0.5rem 0; color: white; font-size: 1.1rem; font-weight: 600;'>UAP Machine Learning 2024</p>
    <p style='margin:0; font-size:0.95rem; color: rgba(255,255,255,0.9); font-weight: 400;'>
        Powered by Deep Learning | CNN Base · MobileNetV2 · EfficientNetB0
    </p>
</div>
""", unsafe_allow_html=True)