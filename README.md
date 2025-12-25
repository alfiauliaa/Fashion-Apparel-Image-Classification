# Fashion Apparel Image Classification

---

## 📋 Table of Contents

1. [Project Description](#-project-description)
   - [Background](#background)
   - [Development Objectives](#development-objectives)
2. [Dataset Source](#-dataset-source)
3. [Preprocessing and Modeling](#-preprocessing-and-modeling)
   - [Data Preparation](#data-preparation)
   - [Data Splitting](#data-splitting)
   - [Model Architecture](#model-architecture)
4. [Installation Steps](#-installation-steps)
   - [Requirements](#requirements)
   - [Running the Application](#running-the-application)
   - [Model Training](#model-training)
5. [Results and Analysis](#-results-and-analysis)
   - [Model Comparison](#model-comparison)
   - [Classification Report](#classification-report)
   - [Training Performance](#training-performance)
6. [Streamlit Web Application](#-streamlit-web-application)
   - [Features](#features)
   - [Application Pages](#application-pages)
7. [Author](#-author)

---

## 📚 Project Description

This project aims to **develop a fashion apparel image classification system** capable of identifying different types of clothing items using Deep Learning approaches. The system employs three different model architectures to compare their performance in classifying fashion items.

### Background

Fashion classification is an important task in e-commerce and retail applications. This project focuses on classifying fashion apparel images into 5 main categories based on clothing types. The classification system uses state-of-the-art deep learning models to achieve high accuracy in identifying fashion items from images.

#### 🔍 **Classification Categories:**

The system classifies fashion items into 5 categories:

- **Dress** 👗: Various styles of dresses
- **Pants** 👖: Trousers and pants
- **Shirt** 👕: Shirts and tops
- **Shoes** 👞: Footwear
- **Shorts** 🩳: Short pants

### Development Objectives

1. **Build Classification Models** to predict fashion apparel categories using three different deep learning architectures
2. **Performance Evaluation**: Compare three models - CNN Base (Non-Pretrained), MobileNetV2 (Transfer Learning), and EfficientNetB0 (Transfer Learning) to determine the best performing model
3. **Deploy Web Application**: Create a Streamlit-based web application for easy access to the classification system

---

## 📊 Dataset Source

The dataset used in this project comes from Kaggle:

### Kaggle Dataset

- **Dataset Title**: _Fashion Apparel Image Classification Dataset_
- **Link**: [Kaggle - Fashion Apparel Dataset](https://www.kaggle.com/datasets/the-data-science-hub/fashion-apparel-images-2024)
- **Description**: This dataset contains fashion apparel images with 10 classes representing different colors and types of clothing items.

#### **Original Dataset Classes:**

The original dataset contains 10 classes with the following distribution:

| Class        | Count     |
| ------------ | --------- |
| black_dress  | 450       |
| black_pants  | 871       |
| black_shirt  | 715       |
| black_shoes  | 766       |
| black_shorts | 328       |
| blue_dress   | 502       |
| blue_pants   | 798       |
| blue_shirt   | 741       |
| blue_shoes   | 523       |
| blue_shorts  | 299       |
| **Total**    | **5,993** |

#### **Modified Dataset for This Project:**

For this project, the dataset was simplified by **removing color distinctions** and merging classes into 5 main categories:

| Class      | Combined From              | Total Images |
| ---------- | -------------------------- | ------------ |
| **Dress**  | black_dress + blue_dress   | 952          |
| **Pants**  | black_pants + blue_pants   | 1,669        |
| **Shirt**  | black_shirt + blue_shirt   | 1,456        |
| **Shoes**  | black_shoes + blue_shoes   | 1,289        |
| **Shorts** | black_shorts + blue_shorts | 627          |
| **Total**  | -                          | **5,993**    |

This modification simplifies the classification task and focuses on identifying clothing types rather than colors, making it more suitable for practical fashion classification applications.

---

## 🧑‍💻 Preprocessing and Modeling

### Data Preparation

The dataset preparation process involves the following steps:

1. **Dataset Merging**:

   - Combined color variants (black and blue) for each clothing type
   - Merged `black_dress` + `blue_dress` → `dress`
   - Merged `black_pants` + `blue_pants` → `pants`
   - Merged `black_shirt` + `blue_shirt` → `shirt`
   - Merged `black_shoes` + `blue_shoes` → `shoes`
   - Merged `black_shorts` + `blue_shorts` → `shorts`

2. **Image Properties**:
   - **Input Size**: 224×224 pixels
   - **Color Channels**: RGB (3 channels)
   - **Format**: JPEG/PNG
   - **Batch Size**: 32

### Data Splitting

The dataset was split into three sets with the following ratio:

| Split              | Percentage | Number of Images |
| ------------------ | ---------- | ---------------- |
| **Training Set**   | 70%        | 4,193 images     |
| **Test Set**       | 20%        | 1,196 images     |
| **Validation Set** | 10%        | 604 images       |

### Model Architecture

Three different models were implemented and compared:

#### 1. **CNN Base (Non-Pretrained)**

- Custom Convolutional Neural Network built from scratch
- **Architecture**:
  - 4 Convolutional blocks (32, 64, 128, 256 filters)
  - BatchNormalization and Dropout (0.3-0.6) after each block
  - MaxPooling2D (2×2) for downsampling
  - Dense layers: 512 → 256 → 5 (output)
  - L2 regularization applied
- **Total Parameters**: 19,400,901
- **Preprocessing**: Simple rescaling (0-1 normalization)

#### 2. **MobileNetV2 (Transfer Learning)**

- Pre-trained on ImageNet dataset
- **Architecture**:
  - MobileNetV2 base (frozen weights)
  - GlobalAveragePooling2D
  - Dense layers: 256 → 128 → 5 (output)
  - BatchNormalization and Dropout (0.5, 0.3, 0.2)
  - L2 regularization applied
- **Total Parameters**: 2,625,605
- **Trainable Parameters**: 364,549
- **Preprocessing**: MobileNetV2 specific preprocessing (-1 to 1 normalization)

#### 3. **EfficientNetB0 (Transfer Learning)**

- Pre-trained on ImageNet dataset
- **Architecture**:
  - EfficientNetB0 base (frozen weights)
  - GlobalAveragePooling2D
  - Dense layers: 256 → 128 → 5 (output)
  - BatchNormalization and Dropout (0.5, 0.3, 0.2)
  - L2 regularization applied
- **Total Parameters**: 4,417,192
- **Trainable Parameters**: 364,549
- **Preprocessing**: EfficientNet specific preprocessing (ImageNet mean/std)

### Training Configuration

- **Optimizer**: Adam
  - CNN Base: learning rate = 0.001
  - MobileNetV2: learning rate = 0.0001
  - EfficientNetB0: learning rate = 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 30
- **Callbacks**:
  - EarlyStopping (monitor='val_loss', patience=10)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (save best model based on val_accuracy)

---

## 🔧 Installation Steps

### Requirements

This project was developed using:

- **Google Colab** (for model training)
- **VSCode** (for Streamlit app development)
- **Python 3.10+**

### Dependencies

All required dependencies are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- `streamlit==1.40.2`
- `tensorflow==2.19.1`
- `pillow==11.0.0`
- `numpy==2.2.1`
- `pandas==2.2.3`
- `matplotlib==3.10.0`

### Running the Application

1. **Clone the repository**:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. **Ensure model files are in place**:

   - The `saved_models_final` folder should contain all model files
   - Required files:
     - `cnn_base.h5` - CNN Base model
     - `mobilenet_trainable.npz` - MobileNet weights
     - `mobilenet_trainable_config.json` - MobileNet config
     - `efficientnet_trainable.npz` - EfficientNet weights
     - `efficientnet_trainable_config.json` - EfficientNet config
     - `metadata.json` - Model metadata
     - `class_indices.json` - Class mappings
     - `*_history.pkl` - Training histories

3. **Run the Streamlit app**:

```bash
streamlit run App.py
```

The application will open in your default web browser at `http://localhost:8501`

### Model Training

If you want to retrain the models:

1. **Open the training notebook in Google Colab**:

   - Upload the provided `.ipynb` notebook file
   - Connect to a GPU runtime for faster training (recommended)

2. **Mount Google Drive**:

   - The notebook is configured to save models to Google Drive
   - Ensure you have sufficient storage space (~100MB)

3. **Dataset Setup**:

   - Upload dataset to Google Drive or download from Kaggle
   - Update dataset path in the notebook

4. **Run all cells**:

   - The notebook will automatically:
     - Load and merge dataset classes
     - Split data (70/20/10)
     - Preprocess images
     - Train all three models
     - Save trained models and training histories
     - Generate evaluation metrics

5. **Download trained models**:
   - After training, download the `saved_models_final` folder
   - Place it in the same directory as `App.py`

---

## 🔍 Results and Analysis

### Model Comparison

Comprehensive comparison of all three models:

| Model              | Test Accuracy | Test Loss  | Parameters | Training Time |
| ------------------ | ------------- | ---------- | ---------- | ------------- |
| **CNN Base**       | 92.22%        | 1.5796     | 19,400,901 | 30 epochs     |
| **MobileNetV2**    | 97.83%        | 0.5064     | 2,625,605  | 30 epochs     |
| **EfficientNetB0** | **98.33%**    | **0.4957** | 4,417,192  | 30 epochs     |

#### **Key Findings:**

1. **Best Overall Performance**: EfficientNetB0

   - Highest test accuracy at **98.33%**
   - Lowest test loss at **0.4957**
   - Excellent generalization with minimal overfitting

2. **Best Efficiency**: MobileNetV2

   - Strong accuracy at **97.83%** (only 0.5% lower than EfficientNetB0)
   - **Smallest model** with only 2.6M parameters
   - Ideal for deployment on resource-constrained devices
   - Fast inference time

3. **Baseline Performance**: CNN Base

   - Respectable **92.22%** accuracy despite no pre-training
   - **Largest model** with 19.4M parameters
   - Demonstrates the value of transfer learning (5-6% accuracy gap)
   - Higher loss indicates less confident predictions

4. **Transfer Learning Advantage**:
   - Pre-trained models (MobileNetV2, EfficientNetB0) significantly outperform CNN Base
   - Achieve 97-98% accuracy vs 92% for base CNN
   - Better convergence and lower loss values
   - Fewer trainable parameters due to frozen base models

### Classification Report

Detailed classification metrics for each model:

#### **CNN Base Model**

| Class            | Precision | Recall | F1-Score | Support  |
| ---------------- | --------- | ------ | -------- | -------- |
| dress            | 0.90      | 0.93   | 0.92     | 190      |
| pants            | 0.94      | 0.93   | 0.94     | 333      |
| shirt            | 0.95      | 0.96   | 0.95     | 291      |
| shoes            | 0.89      | 0.98   | 0.94     | 257      |
| shorts           | 0.91      | 0.69   | 0.78     | 125      |
| **Accuracy**     | -         | -      | **0.92** | **1196** |
| **Macro Avg**    | 0.92      | 0.90   | 0.90     | 1196     |
| **Weighted Avg** | 0.92      | 0.92   | 0.92     | 1196     |

**Analysis:**

- Struggles most with **shorts** (F1: 0.78) - only 69% recall
- Best performance on **shirt** (F1: 0.95)
- Overall balanced precision and recall across most classes

#### **MobileNetV2 Model**

| Class            | Precision | Recall | F1-Score | Support  |
| ---------------- | --------- | ------ | -------- | -------- |
| dress            | 0.98      | 0.97   | 0.97     | 190      |
| pants            | 0.98      | 0.98   | 0.98     | 333      |
| shirt            | 0.98      | 1.00   | 0.99     | 291      |
| shoes            | 0.99      | 0.99   | 0.99     | 257      |
| shorts           | 0.96      | 0.90   | 0.93     | 125      |
| **Accuracy**     | -         | -      | **0.98** | **1196** |
| **Macro Avg**    | 0.98      | 0.97   | 0.97     | 1196     |
| **Weighted Avg** | 0.98      | 0.98   | 0.98     | 1196     |

**Analysis:**

- Excellent performance across all classes (F1: 0.93-0.99)
- **Perfect recall** on shirt category (1.00)
- **Shorts** still most challenging but much improved (F1: 0.93 vs 0.78 in CNN)
- Very high precision (96-99%) indicates few false positives

#### **EfficientNetB0 Model**

| Class            | Precision | Recall | F1-Score | Support  |
| ---------------- | --------- | ------ | -------- | -------- |
| dress            | 0.99      | 0.99   | 0.99     | 190      |
| pants            | 0.97      | 0.98   | 0.98     | 333      |
| shirt            | 0.99      | 0.99   | 0.99     | 291      |
| shoes            | 1.00      | 1.00   | 1.00     | 257      |
| shorts           | 0.95      | 0.92   | 0.93     | 125      |
| **Accuracy**     | -         | -      | **0.98** | **1196** |
| **Macro Avg**    | 0.98      | 0.98   | 0.98     | 1196     |
| **Weighted Avg** | 0.98      | 0.98   | 0.98     | 1196     |

**Analysis:**

- **Perfect performance** on shoes (100% precision and recall)
- Near-perfect scores on dress and shirt (F1: 0.99)
- Most consistent performance across all classes
- Best overall macro average (0.98 in all metrics)

### Performance Comparison by Class

| Class  | CNN Base F1 | MobileNetV2 F1 | EfficientNetB0 F1 | Best Model                 |
| ------ | ----------- | -------------- | ----------------- | -------------------------- |
| dress  | 0.92        | 0.97           | **0.99**          | EfficientNetB0             |
| pants  | 0.94        | 0.98           | **0.98**          | MobileNetV2/EfficientNetB0 |
| shirt  | 0.95        | **0.99**       | **0.99**          | MobileNetV2/EfficientNetB0 |
| shoes  | 0.94        | 0.99           | **1.00**          | EfficientNetB0             |
| shorts | 0.78        | 0.93           | **0.93**          | MobileNetV2/EfficientNetB0 |

**Insights:**

- **Shorts** is consistently the most challenging category across all models

  - Smallest dataset (125 test samples)
  - Likely due to visual similarity with pants
  - Transfer learning helps significantly (0.78 → 0.93)

- **Shoes** benefits most from EfficientNetB0

  - Perfect classification (1.00 F1-score)
  - Distinct visual features make it easiest to classify

- **Transfer learning models** show 5-15% improvement over CNN Base
  - Most dramatic improvement in shorts category (+15%)
  - Consistent gains across all classes

### Training Performance

#### Training Summary

| Model          | Final Train Acc | Final Val Acc | Best Val Acc | Epochs to Best |
| -------------- | --------------- | ------------- | ------------ | -------------- |
| CNN Base       | 96.78%          | 92.88%        | 92.88%       | 30             |
| MobileNetV2    | 98.11%          | 97.52%        | 97.68%       | 29             |
| EfficientNetB0 | 97.70%          | 98.18%        | 98.68%       | 28             |

#### Learning Characteristics

**CNN Base:**

- Gradual improvement over 30 epochs
- Shows signs of overfitting in later epochs (gap between train and val)
- Learning rate reduction helped at epoch 26
- Final validation accuracy plateau at ~93%

**MobileNetV2:**

- Rapid initial convergence (>93% accuracy in epoch 1)
- Stable validation performance throughout
- Minimal overfitting (train-val gap < 1%)
- Consistent improvement until epoch 24
- Transfer learning enables fast learning

**EfficientNetB0:**

- Fastest and most stable convergence
- Best generalization (validation > training in some epochs)
- Minimal overfitting throughout training
- Achieves >96% validation accuracy in epoch 5
- Best overall learning curve

#### Overfitting Analysis

| Model          | Train-Val Gap | Overfitting Level       |
| -------------- | ------------- | ----------------------- |
| CNN Base       | ~4%           | Moderate                |
| MobileNetV2    | ~0.6%         | Minimal                 |
| EfficientNetB0 | ~-0.5%        | None (generalizes well) |

---

## 🎯 Streamlit Web Application

A user-friendly web application built with Streamlit to make the classification system easily accessible.

### Features

The application provides several key features:

1. **Modern Dark Theme Design**

   - Sleek gradient-based UI with purple/blue color scheme
   - Smooth animations and transitions
   - Professional and visually appealing interface

2. **Multiple Model Support**

   - Switch between CNN Base, MobileNetV2, and EfficientNetB0
   - Real-time model selection
   - Displays model-specific information and accuracy

3. **Flexible Image Input**

   - Upload images (JPG, PNG, JPEG)
   - Use camera for real-time capture
   - Automatic image preprocessing

4. **Comprehensive Predictions**

   - Predicted class with confidence percentage
   - Visual confidence indicators
   - Detailed probability distribution for all classes
   - Interactive bar charts

5. **Educational Content**
   - Dataset statistics and visualizations
   - Training history with learning curves
   - Model architecture details
   - Performance comparison tables

### Application Pages

#### **1. 🏠 Home**

Main landing page featuring:

- Overview of the classification system
- Fashion categories with emoji icons
- Key metrics (5 categories, 3 models, best accuracy, total training images)
- Model performance comparison table
- Quick info boxes about features and dataset

#### **2. 📊 Dataset Overview**

Detailed dataset information:

- Dataset distribution statistics
- Sample images from each category (dress, pants, shirt, shoes, shorts)
- Data split visualization (70% train, 20% test, 10% val)
- Bar charts showing image counts per category
- Configuration details (image size, preprocessing methods)

#### **3. 📈 Training History**

Interactive training analysis:

- Model selector dropdown
- Training and validation accuracy curves
- Training and validation loss curves
- Epoch-by-epoch metrics table
- Best performance highlights with epoch information
- Final training statistics

#### **4. 🎯 Model Details**

Technical specifications:

- Architecture type (non-pretrained vs transfer learning)
- Total parameters and trainable parameters
- Test accuracy and loss
- Input shape and preprocessing method
- Layer-by-layer architecture summary
- Training configuration details

#### **5. 🔮 Predict**

Main prediction interface:

- Model selection dropdown with accuracy info
- Upload or camera input options
- Real-time image preview
- AI analysis with loading animation
- Prediction result box showing:
  - Predicted class with emoji icon
  - Confidence percentage
  - Confidence level indicator (Very High/High/Moderate/Low)
- Detailed analysis section:
  - Horizontal bar chart of all class probabilities
  - Complete probability table
  - Top 3 predictions with medals (🥇🥈🥉)
- Tips for best results

### User Interface Highlights

**Design Elements:**

- Gradient backgrounds and cards
- Hover effects and animations
- Color-coded confidence levels
- Responsive layout for all screen sizes
- Interactive charts and visualizations
- Professional typography and spacing

**Usability Features:**

- Clear navigation sidebar
- Intuitive input methods
- Real-time feedback
- Helpful tooltips and instructions
- Error handling and validation
- System information display

---

## 👤 Author

**Fathul Agit Darmawan**

**Academic Information:**

- 📘 NIM: 202110370311169
- 🎓 Program: Teknik Informatika
- 🏛️ Institution: Universitas Muhammadiyah Malang
- 📅 Year: 2024
- 📚 Course: Machine Learning - UAP (Ujian Akhir Praktikum)

**Project Information:**

- 🔬 Project Type: Final Practical Exam Project
- 🎯 Topic: Fashion Apparel Image Classification
- 🤖 Focus: Deep Learning & Transfer Learning
- 📊 Models: CNN, MobileNetV2, EfficientNetB0

---

## 📝 License

This project is created for educational purposes as part of the Machine Learning course final practical exam (UAP) at Universitas Muhammadiyah Malang.

---

## 🙏 Acknowledgments

- **Dataset**: Fashion Apparel Image Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/the-data-science-hub/fashion-apparel-images-2024)
- **Frameworks**: TensorFlow/Keras for model development
- **Web Framework**: Streamlit for application deployment
- **Pre-trained Models**: MobileNetV2 and EfficientNetB0 from Keras Applications
- **Training Platform**: Google Colab for GPU-accelerated training
- **Development Tools**: VSCode and Jupyter Notebook

---

<div align="center">
  <p><b>Fashion Apparel Image Classification</b></p>
  <p>Machine Learning UAP 2024 - Universitas Muhammadiyah Malang</p>
  <p>Built with TensorFlow, Keras & Streamlit</p>
</div>
