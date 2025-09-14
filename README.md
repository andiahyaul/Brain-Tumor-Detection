# Brain Tumor Detection Project

A deep learning project for detecting brain tumors from MRI images using an ensemble model approach, achieving **81.58% accuracy**.

## 📖 Project Overview

This project implements a comprehensive brain tumor detection system using TensorFlow/Keras with an ensemble modeling approach. The system processes MRI brain scan images to classify them as either containing a tumor or being tumor-free.

## 🎯 Results

- **Final Accuracy**: 81.58% on test set
- **Model Architecture**: Weighted ensemble of two trained models
- **Dataset**: 253 MRI brain scan images (155 tumor, 98 no tumor)
- **Optimization**: Custom weighted averaging with threshold tuning

## 📁 Project Structure

```
Tumor Detection/
├── Data/                          # Original dataset
│   ├── yes/                      # Tumor images
│   └── no/                       # No tumor images
├── processed_data/               # Preprocessed datasets
│   ├── train_images.npz
│   ├── val_images.npz
│   ├── test_images.npz
│   └── metadata.json
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
│   ├── Data_Exploration.ipynb   # Dataset analysis
│   ├── Data_Preprocessing.ipynb # Data preparation
│   ├── Baseline_Train.ipynb     # Initial model training
│   ├── Ensemble_Training.ipynb  # Advanced ensemble methods
│   └── Result.ipynb             # Final results
├── src/                         # Python source code
│   ├── ensemble_model.py        # Main ensemble implementation
│   ├── validate_ensemble.py     # Model validation
│   └── deploy_ensemble.py       # Deployment utilities
├── artifacts/                   # Model artifacts
├── docs/                       # Documentation
│   └── requirements.txt
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install -r docs/requirements.txt
```

### Running the Ensemble Model

```python
from src.ensemble_model import BrainTumorEnsemble

# Load pre-trained ensemble
ensemble = BrainTumorEnsemble()

# Make predictions
predictions = ensemble.predict(your_images)

# Evaluate performance
results = ensemble.evaluate(test_images, test_labels)
```

## 📊 Model Performance

| Model Component | Accuracy | Weight |
|----------------|----------|---------|
| Original Model | ~78% | 0.492 |
| Fine-tuned Model | ~79% | 0.508 |
| **Ensemble** | **81.58%** | - |

### Key Features

- **Weighted Ensemble**: Optimized model combination
- **Threshold Tuning**: Custom classification threshold (0.55)
- **Data Augmentation**: Enhanced training dataset
- **Cross-validation**: Robust model evaluation
- **Preprocessing Pipeline**: Automated image preparation

## 🔬 Technical Details

### Dataset Analysis
- **Total Images**: 253 MRI scans
- **Class Distribution**: 61.3% tumor, 38.7% no tumor
- **Image Quality**: Variable resolution, normalized during preprocessing
- **Validation**: Comprehensive data exploration and quality assessment

### Model Architecture
- **Base Models**: CNN architectures with transfer learning
- **Ensemble Method**: Weighted averaging with learned weights
- **Optimization**: Adam optimizer with binary crossentropy loss
- **Evaluation Metrics**: Accuracy, AUC-ROC, precision, recall

### Training Process
1. **Data Exploration**: Comprehensive dataset analysis
2. **Preprocessing**: Image normalization and augmentation
3. **Baseline Training**: Initial model development
4. **Ensemble Creation**: Multiple model combination
5. **Validation**: Performance evaluation and tuning

## 📈 Notebooks Overview

1. **Data_Exploration.ipynb**: Complete dataset analysis and validation
2. **Data_Preprocessing.ipynb**: Image preprocessing and augmentation
3. **Baseline_Train.ipynb**: Initial model training and evaluation
4. **Ensemble_Training.ipynb**: Advanced ensemble techniques
5. **Result.ipynb**: Final results and model comparison

## 🛠️ Usage Examples

### Basic Prediction
```python
import numpy as np
from src.ensemble_model import BrainTumorEnsemble

# Load ensemble
ensemble = BrainTumorEnsemble()

# Load your MRI image (preprocessed)
image = np.load('your_mri_image.npy')

# Get prediction
prediction = ensemble.predict(image.reshape(1, -1))
probability = ensemble.predict(image.reshape(1, -1), return_probabilities=True)

print(f"Prediction: {'Tumor' if prediction[0] == 1 else 'No Tumor'}")
print(f"Confidence: {probability[0][0]:.2%}")
```

### Model Evaluation
```python
# Load test data
test_images, test_labels = load_test_data()

# Comprehensive evaluation
results = ensemble.evaluate(test_images, test_labels)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"AUC Score: {results['auc']:.4f}")
```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- OpenCV
- Pillow

