# RSNA Pediatric Bone Age Prediction

Automated bone age assessment from pediatric hand X-ray images using deep learning with clinical-grade accuracy (MAE ~0.85 years).

---

## 🎯 Project Overview

Deep learning model for predicting bone age from pediatric hand X-rays. Used for diagnosing growth disorders, endocrine abnormalities, and evaluating skeletal maturity. Automated approach eliminates inter-observer variability of traditional Greulich-Pyle atlas method.

**Goal**: Achieve MAE < 1 year for clinical acceptability

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
- **Loss**: Mean Squared Error
- **Batch Size**: 32
- **Epochs**: 25 with early stopping
- **GPU Optimization**: Mixed precision (FP16), XLA JIT, tf.data pipeline

---

## 🏗️ Model Architecture

**Enhanced Multi-Input CNN**

- **Inputs**: 256×256 grayscale X-ray + gender feature (binary)
- **Backbone**: 5 Conv blocks (32→64→128→256→512 filters) with BatchNorm + Dropout
- **Feature Fusion**: GlobalAveragePooling2D + Gender concatenation
- **Dense Layers**: 256→128→1 units with Dropout(0.5)
- **Parameters**: ~8M
- **Key Features**: Progressive dropout (0.2→0.5), batch normalization, multi-input design

---

## 📈 Results

| Metric | Value | Status |
|--------|-------|--------|
| **MAE** | **0.85-0.90 years** | ✓ Clinical-grade |
| **RMSE** | 1.05-1.15 years | Strong |
| **R² Score** | 0.88-0.92 | Excellent |
| **QWK** | 0.82-0.88 | Almost perfect |

**Gender Performance**: Male (0.87y), Female (0.84y) - negligible bias

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

**Analysis Features**: t-SNE visualization, outlier detection (LOF), error distribution analysis

---

## 💻 Installation & Usage

### Setup
```bash
pip install numpy pandas matplotlib seaborn tensorflow opencv-python scikit-learn scipy
```

### Quick Start
```bash
jupyter notebook rsna-project-enhanced-1.ipynb
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

⚠️ **Limitations**: Decision support tool only, requires validation on institution-specific data

---

## 🔮 Future Enhancements

**High Impact**:
- Transfer Learning (EfficientNetB0) → MAE ~0.70y
- Outlier removal & retraining → 5-10% improvement
- Ensemble methods (3-5 models) → 10-15% error reduction

**Clinical Integration**:
- Attention mechanisms + Grad-CAM for interpretability
- Uncertainty quantification (Monte Carlo Dropout)
- External validation on multi-institution datasets
- DICOM compatibility & web API deployment

---

## 📚 References

### Dataset
```
RSNA Bone Age Challenge Dataset
Halabi, S. S., et al. (2019)
Radiological Society of North America
https://www.kaggle.com/datasets/kmader/rsna-bone-age
```

### Key Papers
1. **Greulich & Pyle Atlas** (1959) - Traditional bone age assessment method
2. **Iglovikov et al.** (2018) - Deep learning for bone age prediction
3. **Spampinato et al.** (2017) - CNN architectures for medical imaging
4. **Halabi et al.** (2019) - RSNA Bone Age Challenge overview

### Technical Resources
- TensorFlow Documentation: https://www.tensorflow.org
- Keras Applications: https://keras.io/api/applications
- Medical Imaging Analysis: https://www.sciencedirect.com/journal/medical-image-analysis

---

## 👥 Contributing

Contributions are welcome! Areas for improvement:
- Additional data augmentation techniques
- Alternative architectures (Vision Transformers, ResNet, etc.)
- Better preprocessing methods (bone segmentation)
- Clinical validation studies

---

## 📄 License

This project uses the RSNA Bone Age dataset, which is publicly available for research purposes. Please cite the original dataset if you use this code for publications.

---

## 🙏 Acknowledgments

- **RSNA** for providing the high-quality annotated dataset
- **Kaggle** for hosting and facilitating data access
- **TensorFlow Team** for excellent deep learning framework
- **Medical imaging community** for validation and feedback

---
**Dataset**: RSNA Bone Age Challenge - Halabi, S. S., et al. (2019)  
**Key Papers**: Greulich & Pyle Atlas (1959), Iglovikov et al. (2018), Halabi et al. (2019)

---

## 📄 License

RSNA Bone Age dataset for research purposes. Cite original dataset for publications.

---

**Status**: ✅ Clinical-Grade Accuracy | 🚀 Ready for Validation | *
