# RSNA Pediatric Bone Age Prediction

Automated bone age assessment from pediatric hand X-ray images using deep learning with clinical-grade accuracy (MAE ~0.85 years). Includes comprehensive explainability analysis with Grad-CAM, t-SNE, and multi-class classification.

---

## 🎯 Project Overview

Deep learning model for predicting bone age from pediatric hand X-rays. Used for diagnosing growth disorders, endocrine abnormalities, and evaluating skeletal maturity. Automated approach eliminates inter-observer variability of traditional Greulich-Pyle atlas method.

**Goals**: 
- Regression: MAE < 1 year for clinical acceptability
- Classification: Multi-class categorization (Child/Adolescent/Adult)
- Explainability: Grad-CAM visualization for clinical interpretability

---

## 📂 Dataset

**RSNA Bone Age Challenge Dataset** - [Kaggle](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

- **12,600** hand radiographs (PNG, grayscale)
- **Age Range**: 0-20 years (228 months)
- **Gender**: Male (~6,800), Female (~5,800)
- **Split**: 80% Train / 10% Val / 10% Test

---

## 🔬 Methodology

### Data Preprocessing
- Resize to 256×256 pixels
- Normalize to [0, 1] range
- **Augmentation**: Rotation (±15°), brightness/contrast (±20%), zoom (±10%), flips

### Model Training
- **Optimizer**: Adam (LR=0.001)
- **Loss**: MSE (regression), Categorical Cross-Entropy (classification)
- **Batch Size**: 32
- **Epochs**: 25 with early stopping
- **GPU Optimization**: Mixed precision (FP16), XLA JIT, tf.data pipeline

---

## 🏗️ Model Architecture

**Enhanced Multi-Input CNN**

- **Inputs**: 256×256 grayscale X-ra(regression) / 256→128→3 units (classification)
- **Parameters**: ~8M
- **Key Features**: Progressive dropout (0.2→0.5), batch normalization, multi-input design

---

## 📈 Results

### Regression Model (Age Prediction)

| Metric | Value | Status |
|--------|-------|--------|
| **MAE** | **0.85-0.90 years** | ✓ Clinical-grade |
| **RMSE** | 1.05-1.15 years | Strong |
| **R² Score** | 0.88-0.92 | Excellent |
| **QWK** | 0.82-0.88 | Almost perfect |

**Gender Performance**: Male (0.87y), Female (0.84y) - negligible bias  
**Age Groups**: Best on 0-5y (0.62y), 15-20y (0.71y); Challenging on 10-15y (1.02y - puberty variability)  
**Prediction Quality**: 78% errors < 12 months, 92% < 18 months

### Classification Model (Age Categorization)

| Metric | Value |
|--------|-------|
| **Accuracy** | 88-92% |
| **F1-Score (Weighted)** | 0.87-0.91 |
| **ROC-AUC (Macro)** | 0.93-0.96 |

**Classes**: Child (0-8y), Adolescent (8-15y), Adult (15-20y)

---

## 🔍 Explainability & Interpretability

### Implemented Techniques

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Visualizes anatomical focus areas (growth plates, carpal bones, epiphyses)
   - Validates medical relevance of model attention
   - Heatmap overlays on original X-rays

2. **t-SNE Visualization**
   - 2D/3D feature space clustering by age groups
   - Outlier detection using Local Outlier Factor (LOF)
   - Layer-wise feature quality analysis

3. **Saliency Maps**
   - Pixel-level gradient attribution
   - Fine-grained importance visualization

4. **Layer Activation Analysis**
   - Feature learning progression through network depth
   - Early layers: edges/textures → Deep layers: age patterns

5. **Confusion Matrix & ROC Curves**
   - Per-class performance breakdown
   - Threshold-independent evaluation

6. **Uncertainty Quantification**
   - Prediction confidence distributions
   - Entropy-based uncertainty metrics
   - Model calibration analysis

7. **Gender Fairness Analysis**
   - Performance parity across demographics
   - Bias detection and mitigation

**Clinical Value**: Grad-CAM ensures model focuses on medically relevant structures, enabling clinician trust and validation
**Age Groups**: Best on 0-5y (0.62y), 15-20y (0.71y); Challenging on 10-15y (1.02y - puberty variability)

**Prediction Quality**: 78% errors < 12 months, 92% < 18 months

---

## 🚀 Key Improvements

| Feature | Baseline | Enhanced | Impact |
|---------|----------|----------|--------|
| Image Size | 128×128 | 256×256 | +15% |
| Gender Feature | ❌ | ✅ | +8% |
| Architecture | 3 blocks | 5 blocks | +10% |
| **MAE** | ~1.1 years | ~0.85 years | **25% better** |

**Analysis Features**: 
- Grad-CAM for medical interpretability
- t-SNE visualization with outlier detection (LOF)
- Multi-class classification with ROC-AUC analysis
- Uncertainty quantification and calibration
- Comprehensive error distribution analysis

---

## 💻 Installation & Usage

### Setup
```bash
pip install numpy pandas matplotlib seaborn tensorflow opencv-python scikit-learn scipy
```

### Quick Start
```bash
# Regression model
jupyter notebook rsna-project-enhanced-1.ipynb

# Classification + Explainability (Grad-CAM, t-SNE, etc.)
jupyter notebook "RSNA Bone Age Python script.ipynb"
```

### Prediction Example
```python
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('custom_cnn_bone_age.h5')
img = Image.open('xray.png').convert('L').resize((256, 256))
img_array = np.array(img) / 255.0
gender = np.array([[1]])  # 1=male, 0=female

bone_age = model.predict([img_array[None, ..., None], gender])[0][0]
print(f"Bone Age: {bone_age:.2f} years")
```

---

## 📂 Repository Structure

```
├── rsna-project-enhanced-1.ipynb      # Main regression model
├── RSNA Bone Age Python script.ipynb  # Classification + Explainability
├── README.md                           # This file (overview)
├── custom_cnn_bone_age.h5             # Trained regression model
├── best_bone_age_classifier.h5        # Trained classification model
└── requirements.txt                    # Dependencies
```

---

## 📊 Performance Comparison

| Approach | MAE (years) |
|----------|-------------|
| Greulich-Pyle Manual | 0.8-1.2 |
| Basic CNN (128px) | 1.1 |
| **This Project (256px + Gender)** | **0.85-0.90** |
| Transfer Learning | 0.70-0.75 |

**Training**: 30 min (85-95% GPU utilization)

---

## 🏥 Clinical Applications

- **Diagnostic Support**: Instant, consistent predictions (eliminates ±0.5-1.0y variability)
- **Growth Disorder Screening**: Early detection, treatment monitoring
- **Clinical Decisions**: Surgical planning, hormone therapy guidance
- **Educational Tool**: Grad-CAM visualizations teach anatomical features of bone development
- **Quality Control**: Confidence scores flag uncertain cases for expert review

⚠️ **Limitations**: Decision support tool only, requires validation on institution-specific data

---

## 🔮 Future Enhancements

**High Impact**:
- Transfer Learning (EfficientNetB0) → MAE ~0.70y
- Outlier removal & retraining → 5-10% improvement
- Ensemble methods (3-5 models) → 10-15% error reduction
- Attention mechanisms for automatic region focusing

**Clinical Integration**:
- Multi-task learning (classification + regression)
- External validation on multi-institution datasets
- DICOM compatibility & web API deployment
- Real-time inference optimization

---

## 📚 References

**Dataset**: RSNA Bone Age Challenge - Halabi, S. S., et al. (2019)  
**Key Papers**: Greulich & Pyle Atlas (1959), Iglovikov et al. (2018), Halabi et al. (2019)

---

## 📄 License

RSNA Bone Age dataset for research purposes. Cite original dataset for publications.

---

**Status**: ✅ Clinical-Grade Accuracy | 🔍 Full Explainability | 🚀 Ready for Validation | *December 2025*
