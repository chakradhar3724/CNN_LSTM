# CAPTCHA Recognition using CNN–LSTM Using PyTorch

This project implements an **end-to-end CAPTCHA recognition system** using a hybrid **Convolutional Neural Network (CNN)** and **Bidirectional LSTM (Long Short-Term Memory)** architecture. The model is trained to recognize **5-character alphanumeric CAPTCHA images** and achieves **high character-level and full CAPTCHA accuracy**.

This project was built as a hands-on deep learning experiment to understand **sequence modeling from images**, inspired by research-level CAPTCHA solvers.

---

## 🔍 Project Overview

Traditional OCR systems struggle with distorted and noisy CAPTCHA images. This project addresses the challenge by:

- Using **CNN layers** to extract spatial visual features  
- Converting image features into sequences  
- Applying **Bidirectional LSTM** layers to model character dependencies  
- Predicting characters **without explicit character segmentation**

---

## 🧠 Model Architecture

The model follows a **CNN + LSTM** pipeline:

### CNN Feature Extractor
- 3 Convolutional blocks  
- Batch Normalization for stability  
- ReLU activations  
- MaxPooling for spatial reduction  
- Dropout for regularization  

### Sequence Modeling
- Feature maps are reshaped along the width dimension  
- Passed to a **2-layer Bidirectional LSTM**  
- Fully connected layer predicts character probabilities at each timestep  

**Architecture Flow:**
```
Input Image → CNN → Feature Sequence → BiLSTM → Fully Connected → Character Predictions
```

---

## 📁 Dataset

The dataset consists of CAPTCHA images where the **filename represents the ground-truth label**.

You can download the dataset used in this project here:
[Download Captcha Dataset](https://drive.google.com/file/d/1Yu7Zli6tW9s5P0e5DtdCDuhhMi-qXViV/view?usp=sharing)

### Dataset Specifications
- Image Size: `200 × 50`
- Color Space: Grayscale
- CAPTCHA Length: `5 characters`
- Character Set: `0–9`, `A–Z`
- Total Classes: `36`

### Dataset Structure
```
captcha_data/
│── A9K3P.png
│── Z4M8Q.jpg
│── ...
```

> ⚠️ **Important Notes**
> - All images must be placed inside the `captcha_data/` directory  
> - Filenames must exactly match the CAPTCHA text  
> - Invalid or unreadable images are automatically skipped  

---

## 🔀 Dataset Split

The dataset is split using `train_test_split`:
- **Training Set:** 90%
- **Validation Set:** 10%

A fixed random seed ensures reproducibility.

---

## ⚙️ Preprocessing Pipeline

Each image undergoes the following preprocessing steps:

- Convert to grayscale  
- Resize to `200 × 50`  
- Gaussian blur to reduce noise  
- Normalize pixel values to `[0, 1]`  
- Add channel dimension for CNN input  

---

## 🚀 Training Details

- Framework: **PyTorch**
- Optimizer: **Adam**
- Learning Rate: `1e-3`
- Loss Function: **CrossEntropyLoss**
- Scheduler: **ReduceLROnPlateau**
- Epochs: `20`
- Batch Size: `64`

The best-performing model is saved automatically during training.

---

## 📊 Evaluation Metrics

Two metrics are used:

1. **Character Accuracy**  
   Percentage of correctly predicted characters across all CAPTCHAs  

2. **CAPTCHA Accuracy**  
   Percentage of CAPTCHAs where **all 5 characters** are predicted correctly  

---

## ✅ Results

| Metric | Performance |
|------|------------|
| Character Accuracy | ~99% |
| CAPTCHA Accuracy | ~95% |

The model converges quickly and maintains stable validation performance.

---

## 📈 Training Curves

The project visualizes:
- Training loss vs epochs  
- Character accuracy vs epochs  
- CAPTCHA accuracy vs epochs  

These plots help verify convergence and detect overfitting.

---

## 🧪 Inference Example

Once trained, the model can predict CAPTCHA text from a new image:

```python
predict_captcha("captcha_data/A9K3P.png")
```

## 🧩 Key Highlights

- No manual character segmentation  
- End-to-end sequence prediction  
- Robust against noise and distortions  
- Research-inspired CNN–LSTM architecture  
- High real-world CAPTCHA accuracy  

---

## 🔮 Future Improvements

- Add CTC Loss for variable-length CAPTCHAs  
- Support lowercase characters and symbols  
- Data augmentation for better generalization  
- Web-based inference demo  
- Transformer-based sequence modeling  

---

## 👤 Author

**Chakradhar Peddavenkatagari**  

Aspiring AI Engineer

Masters in Computer Science

The State University of New York at Buffalo 
