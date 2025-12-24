"""
Test Model Loading - Keras 3 Compatible
Run this FIRST before starting Streamlit
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import json
import os

print("=" * 60)
print("TESTING MODEL LOADING - KERAS 3")
print("=" * 60)

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Check Keras version
keras_major = int(tf.keras.__version__.split('.')[0])
print(f"Keras Major Version: {keras_major}")

if keras_major < 3:
    print("\n⚠️  WARNING: You have Keras 2 (TF < 2.16)")
    print("   Models were saved with Keras 3")
    print("   Please upgrade: pip install --upgrade tensorflow")
    exit(1)

# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: CHECKING FILES")
print("=" * 60)

model_dir = Path("saved_models")

if not model_dir.exists():
    print("\n❌ ERROR: 'saved_models' folder not found!")
    print("   Make sure you:")
    print("   1. Downloaded saved_models2 from Google Drive")
    print("   2. Renamed it to 'saved_models'")
    print("   3. Placed it in the same folder as this script")
    exit(1)

print("\n✅ 'saved_models' folder found")

# Check metadata
required_files = ['metadata.json', 'class_indices.json']
for file in required_files:
    if (model_dir / file).exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} MISSING!")

# Check model files - Keras 3 format
model_files = {
    'CNN Base': ['cnn_base.keras', 'cnn_base.h5', 'cnn_base.weights.h5'],
    'MobileNetV2': ['mobilenet.keras', 'mobilenet.h5', 'mobilenet.weights.h5'],
    'EfficientNetB0': ['efficientnet.keras', 'efficientnet.h5', 'efficientnet.weights.h5']
}

print("\n📁 Model Files:")
for model_name, files in model_files.items():
    print(f"\n  {model_name}:")
    found_any = False
    for file in files:
        path = model_dir / file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    ✅ {file} ({size_mb:.1f} MB)")
            found_any = True
        else:
            print(f"    ❌ {file}")
    
    if not found_any:
        print(f"    ⚠️  WARNING: No files found for {model_name}")

# ============================================================
# 2. LOAD METADATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: LOADING METADATA")
print("=" * 60)

try:
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(model_dir / "class_indices.json", 'r') as f:
        class_indices = json.load(f)
    
    print("\n✅ Metadata loaded successfully")
    print(f"  Image Size: {metadata['img_size']}")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Class Names: {', '.join(metadata['classes'])}")
    print(f"  Saved with TF: {metadata.get('tensorflow_version', 'Unknown')}")
    print(f"  Saved with Keras: {metadata.get('keras_version', 'Unknown')}")
    
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    
except Exception as e:
    print(f"\n❌ Error loading metadata: {e}")
    exit(1)

# ============================================================
# 3. TEST MODEL LOADING - KERAS 3
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: TESTING MODEL LOADING (Keras 3)")
print("=" * 60)

def test_load_model(model_name, display_name):
    """Test loading model with Keras 3"""
    
    print(f"\n🔬 Testing {display_name}...")
    print("-" * 40)
    
    success = False
    loaded_model = None
    load_method = None
    
    # Strategy 1: .keras (Native Keras 3 - BEST)
    keras_path = model_dir / f"{model_name}.keras"
    if keras_path.exists():
        print(f"  Trying {model_name}.keras...")
        try:
            loaded_model = tf.keras.models.load_model(
                str(keras_path),
                compile=False
            )
            print("    ✅ SUCCESS with .keras")
            success = True
            load_method = ".keras"
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:100]}")
    
    # Strategy 2: .h5 with safe_mode=False
    if not success:
        h5_path = model_dir / f"{model_name}.h5"
        if h5_path.exists():
            print(f"  Trying {model_name}.h5...")
            try:
                loaded_model = tf.keras.models.load_model(
                    str(h5_path),
                    compile=False,
                    safe_mode=False
                )
                print("    ✅ SUCCESS with .h5")
                success = True
                load_method = ".h5"
            except Exception as e:
                print(f"    ❌ Failed: {str(e)[:100]}")
    
    if success:
        print(f"\n  ✅ {display_name} loaded successfully via {load_method}")
        print(f"  📊 Total parameters: {loaded_model.count_params():,}")
        
        # Quick shape test
        try:
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            output = loaded_model.predict(test_input, verbose=0)
            print(f"  ✅ Model inference OK - Output shape: {output.shape}")
        except Exception as e:
            print(f"  ⚠️  Inference test failed: {str(e)[:100]}")
        
        return loaded_model, True
    else:
        print(f"\n  ❌ {display_name} failed to load")
        return None, False

# Test each model
results = {}

models_to_test = [
    ('cnn_base', 'CNN Base'),
    ('mobilenet', 'MobileNetV2'),
    ('efficientnet', 'EfficientNetB0')
]

for model_name, display_name in models_to_test:
    model, success = test_load_model(model_name, display_name)
    results[display_name] = {
        'model': model,
        'success': success
    }

# ============================================================
# 4. TEST PREDICTIONS WITH PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: TESTING PREDICTIONS")
print("=" * 60)

# Create test image
print("\n🖼️  Creating test image...")
test_image = Image.new('RGB', (224, 224), color='red')
print("  ✅ Test image created (224x224, red)")

def preprocess_image(image, model_name):
    """Preprocess image for model"""
    img_array = np.array(image).astype(np.float32)
    
    if model_name == 'CNN Base':
        img_array = img_array / 255.0
    elif model_name == 'MobileNetV2':
        img_array = img_array / 127.5 - 1.0
    elif model_name == 'EfficientNetB0':
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        img_array = (img_array - mean) / std
    
    return np.expand_dims(img_array, axis=0)

# Test predictions
for model_name, result in results.items():
    if result['success']:
        print(f"\n🔮 Testing prediction with {model_name}...")
        try:
            model = result['model']
            processed = preprocess_image(test_image, model_name)
            
            predictions = model.predict(processed, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            pred_class = class_names[pred_idx]
            
            print(f"  ✅ Prediction successful!")
            print(f"     Predicted: {pred_class}")
            print(f"     Confidence: {confidence*100:.2f}%")
            print(f"     All probabilities: {[f'{p:.3f}' for p in predictions[0]]}")
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {str(e)[:150]}")

# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

successful = sum(1 for r in results.values() if r['success'])
total = len(results)

print(f"\n📊 Models Loaded: {successful}/{total}")

for model_name, result in results.items():
    status = "✅ READY" if result['success'] else "❌ FAILED"
    print(f"  {status} - {model_name}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if successful == 0:
    print("\n❌ CRITICAL: No models loaded!")
    print("\n🔧 Solutions:")
    print("1. Check TensorFlow version (need 2.16+)")
    print("2. Verify model files exist")
    print("3. Try re-downloading from Colab")
    
elif successful < total:
    print(f"\n⚠️  WARNING: Only {successful}/{total} models loaded")
    print("\n💡 You can still run Streamlit with available models")
    print("   Failed models will be disabled in the app")
    
else:
    print("\n✅ EXCELLENT: All models loaded successfully!")
    print("\n🚀 Ready to run Streamlit:")
    print("   streamlit run app.py")
    print("\n📋 Models available:")
    for model_name in results.keys():
        print(f"   • {model_name}")

print("\n" + "=" * 60)
print("ENVIRONMENT INFO")
print("=" * 60)

print(f"\n✅ Your Environment:")
print(f"  TensorFlow: {tf.__version__}")
print(f"  Keras: {tf.keras.__version__}")
print(f"  Python: {tf.__version__}")

print(f"\n✅ Model Environment (from metadata):")
print(f"  TensorFlow: {metadata.get('tensorflow_version', 'Unknown')}")
print(f"  Keras: {metadata.get('keras_version', 'Unknown')}")

if successful == total:
    print("\n✅ Versions compatible - all models loaded!")
else:
    print("\n⚠️  Some models failed - check errors above")

print("\nTest complete!")