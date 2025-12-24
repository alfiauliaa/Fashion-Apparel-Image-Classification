"""
Verify Weights - Test apakah weights benar-benar dimuat
Jalankan ini untuk debug prediksi
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import json

print("=" * 60)
print("VERIFY WEIGHTS & PREDICTIONS")
print("=" * 60)

# Load metadata
model_dir = Path("saved_models_fixed")

with open(model_dir / "metadata.json", 'r') as f:
    metadata = json.load(f)
with open(model_dir / "class_indices.json", 'r') as f:
    class_indices = json.load(f)

class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
print(f"\nClasses: {class_names}")

# ============================================================
# TEST 1: Cek apakah weights file ada dan bisa dibaca
# ============================================================

print("\n" + "=" * 60)
print("TEST 1: Check Weights Files")
print("=" * 60)

for model_name in ['mobilenet', 'efficientnet']:
    npz_path = model_dir / f'{model_name}_top_weights.npz'
    json_path = model_dir / f'{model_name}_weights_info.json'
    
    print(f"\n{model_name.upper()}:")
    
    if npz_path.exists():
        print(f"  ✅ NPZ file exists: {npz_path.name}")
        
        # Load dan cek isi
        data = np.load(npz_path)
        print(f"  📊 Number of weight arrays: {len(data.files)}")
        print(f"  📋 Keys (first 5): {data.files[:5]}")
        
        # Cek size
        total_params = sum(data[k].size for k in data.files)
        print(f"  💾 Total parameters: {total_params:,}")
    else:
        print(f"  ❌ NPZ file NOT found!")
    
    if json_path.exists():
        print(f"  ✅ JSON file exists")
        with open(json_path, 'r') as f:
            info = json.load(f)
        print(f"  📋 Layers saved: {len(info['layer_info'])}")
    else:
        print(f"  ❌ JSON file NOT found!")

# ============================================================
# TEST 2: Compare weights before and after loading
# ============================================================

print("\n" + "=" * 60)
print("TEST 2: Compare Weights Before/After Loading")
print("=" * 60)

from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2

# Build MobileNet
print("\n📦 Building MobileNetV2...")
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
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
    layers.Dense(5, activation='softmax')
])

# Get initial weights dari top layers
print("\n📊 Initial weights (ImageNet only):")
top_layers_before = []
for i, layer in enumerate(model.layers):
    if i > 0 and len(layer.get_weights()) > 0:  # Skip base model
        weights = layer.get_weights()
        top_layers_before.append({
            'layer': layer.name,
            'num_weights': len(weights),
            'shapes': [w.shape for w in weights],
            'means': [float(np.mean(w)) for w in weights],
            'stds': [float(np.std(w)) for w in weights]
        })
        print(f"  {layer.name}: {len(weights)} arrays, mean={np.mean([np.mean(w) for w in weights]):.4f}")

# Load weights
print("\n🔄 Loading trained weights...")
npz_path = model_dir / 'mobilenet_top_weights.npz'
json_path = model_dir / 'mobilenet_weights_info.json'

if npz_path.exists() and json_path.exists():
    weights_data = np.load(npz_path)
    with open(json_path, 'r') as f:
        weights_info = json.load(f)
    
    # Load weights
    base_found = False
    layer_idx = 0
    loaded_count = 0
    
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_found = True
            continue
        
        if base_found and len(layer.get_weights()) > 0:
            stored_layers = list(weights_info['layer_info'].keys())
            if layer_idx < len(stored_layers):
                stored_layer_name = stored_layers[layer_idx]
                num_weights = len(layer.get_weights())
                
                weights_to_load = []
                for i in range(num_weights):
                    key = f"{stored_layer_name}_w{i}"
                    if key in weights_data:
                        weights_to_load.append(weights_data[key])
                
                if len(weights_to_load) == num_weights:
                    try:
                        layer.set_weights(weights_to_load)
                        loaded_count += 1
                        print(f"  ✅ Loaded: {layer.name}")
                    except Exception as e:
                        print(f"  ❌ Failed: {layer.name} - {str(e)[:50]}")
            
            layer_idx += 1
    
    print(f"\n✅ Loaded weights for {loaded_count} layers")
else:
    print("\n❌ Weights files not found!")

# Get weights after loading
print("\n📊 After loading trained weights:")
top_layers_after = []
for i, layer in enumerate(model.layers):
    if i > 0 and len(layer.get_weights()) > 0:
        weights = layer.get_weights()
        top_layers_after.append({
            'layer': layer.name,
            'num_weights': len(weights),
            'shapes': [w.shape for w in weights],
            'means': [float(np.mean(w)) for w in weights],
            'stds': [float(np.std(w)) for w in weights]
        })
        print(f"  {layer.name}: {len(weights)} arrays, mean={np.mean([np.mean(w) for w in weights]):.4f}")

# Compare
print("\n🔍 Comparison (Before vs After):")
if len(top_layers_before) == len(top_layers_after):
    weights_changed = 0
    for before, after in zip(top_layers_before, top_layers_after):
        mean_diff = abs(np.mean(before['means']) - np.mean(after['means']))
        if mean_diff > 0.01:  # Significant change
            print(f"  ✅ {before['layer']}: CHANGED (diff={mean_diff:.4f})")
            weights_changed += 1
        else:
            print(f"  ⚠️  {before['layer']}: NO CHANGE")
    
    if weights_changed == 0:
        print("\n❌ WARNING: No weights changed! Weights not loaded properly!")
    else:
        print(f"\n✅ {weights_changed}/{len(top_layers_before)} layers changed")

# ============================================================
# TEST 3: Test Prediction with Sample Image
# ============================================================

print("\n" + "=" * 60)
print("TEST 3: Test Prediction")
print("=" * 60)

# Create test image (dummy)
print("\n🖼️  Creating test image (red square)...")
test_img = np.ones((224, 224, 3), dtype=np.float32) * 255.0
test_img[:, :, 1:] = 0  # Make it red

# Preprocess
test_img_processed = test_img / 127.5 - 1.0
test_img_batch = np.expand_dims(test_img_processed, axis=0)

# Predict
print("\n🔮 Making prediction...")
predictions = model.predict(test_img_batch, verbose=0)
pred_idx = np.argmax(predictions[0])
confidence = predictions[0][pred_idx]
pred_class = class_names[pred_idx]

print(f"\n📊 Prediction Results:")
print(f"   Predicted: {pred_class}")
print(f"   Confidence: {confidence*100:.2f}%")
print(f"\n   All probabilities:")
for i, cls in enumerate(class_names):
    print(f"      {cls}: {predictions[0][i]*100:.2f}%")

# ============================================================
# TEST 4: Test with Real Sample Image (if available)
# ============================================================

print("\n" + "=" * 60)
print("TEST 4: Test with Real Sample (if available)")
print("=" * 60)

sample_dir = model_dir / 'sample_images'
if sample_dir.exists():
    # Coba ambil sample dari setiap kelas
    for cls in class_names[:2]:  # Test 2 kelas saja
        cls_dir = sample_dir / cls
        if cls_dir.exists():
            images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
            if images:
                img_path = images[0]
                print(f"\n📸 Testing with {cls} image: {img_path.name}")
                
                # Load dan preprocess
                img = Image.open(img_path).resize((224, 224))
                img_array = np.array(img).astype(np.float32)
                
                # Preprocess untuk MobileNet
                img_array = img_array / 127.5 - 1.0
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Predict
                pred = model.predict(img_batch, verbose=0)
                pred_idx = np.argmax(pred[0])
                pred_class = class_names[pred_idx]
                confidence = pred[0][pred_idx]
                
                print(f"   True class: {cls}")
                print(f"   Predicted: {pred_class}")
                print(f"   Confidence: {confidence*100:.2f}%")
                
                if pred_class == cls:
                    print(f"   ✅ CORRECT!")
                else:
                    print(f"   ❌ WRONG!")
else:
    print("\n⚠️  sample_images folder not found, skipping real image test")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 60)

print("\n🔍 Diagnosis:")
if weights_changed > 0:
    print("  ✅ Weights are being loaded")
    print("  ⚠️  But predictions might still be wrong because:")
    print("     1. Weights mungkin tidak match sempurna dengan arsitektur")
    print("     2. Preprocessing mungkin berbeda")
    print("     3. Model perlu fine-tuning lagi")
else:
    print("  ❌ Weights NOT loaded properly!")
    print("  💡 Solution: Re-save weights di Colab dengan cara berbeda")

print("\n💡 Recommendations:")
print("  1. Pastikan arsitektur di Colab dan local IDENTIK")
print("  2. Cek preprocessing - harus sama persis")
print("  3. Pertimbangkan untuk save full model (bukan hanya weights)")

print("\n" + "=" * 60)