# -OralScan-AI-Oral-Cancer-Detection-System

🚀 OralScan AI is a deep learning–powered system that helps in the early detection of oral cancer using image analysis.
It uses Convolutional Neural Networks (CNNs) and Explainable AI (XAI) techniques (Grad-CAM) to not only predict if an image shows cancer but also highlight the exact suspicious regions using heatmaps.

## ✨ Introduction
- Oral cancer is one of the leading causes of cancer-related deaths worldwide.
- Early detection = Higher survival rate ✅.

## This project uses deep learning models to classify oral images into:
  -Cancerous 🟥
  -Non-Cancerous 🟩
- It also provides heatmaps 🔥 to visually explain where cancerous regions are located in the oral cavity.

## 🛠️ System Architecture
- Frontend: Simple web interface / Colab interface to upload images.
- Backend: Python + TensorFlow/Keras models.

## Models:
- Model 1 (VGG16-based) – high accuracy, large-scale training.
- Model 2 (Lightweight CNN) – fast, optimized for real-time screening.

- XAI Integration: Grad-CAM heatmaps for interpretability.
- Output: Single PNG image containing prediction + heatmap.

## 🧠 Algorithms & Key Concepts
- Deep Learning CNNs:
  - VGG16 (Transfer Learning): fine-tuned on oral cancer dataset.
  - Custom CNN: lightweight and fast for edge devices.

## Data Augmentation:
- Patch Shuffle 🧩 – random shuffle to improve generalization.
- CutMix ✂️ – mixes images for robust learning.

## Explainable AI (XAI):
- Grad-CAM highlights regions responsible for prediction.
- Overlayed heatmap helps doctors see why the model decided “cancer”.

## 📂 Input & Output
## 🔹 Input:
  -Oral cavity image (e.g., .jpg, .png)

## 🔹 Output:
## PNG image with:
✅ Original Image
✅ Grad-CAM Heatmap overlay
✅ Prediction (Cancer / Non-Cancer)
✅ Confidence Score (%) displayed in corner

## 📊 Model Performance
- Model 1 – VGG16 Large Model
      - Accuracy: 89.73%
      - Precision: 88.5%
      - Recall (Sensitivity): 90.1%
      - F1-Score: 89.3%
      - Inference time: ~2.1s / image

- Model 2 – Lightweight CNN
        - Accuracy: 87.54%
        - Precision: 85.9%
        - Recall: 88.2%
        - F1-Score: 87.0%
        - Inference time: ~1.3s / image

## ⚙️ Requirements
      - Python 3.8+
      - TensorFlow / Keras 🧠
      - NumPy, OpenCV, Matplotlib
      - scikit-learn (for evaluation)

## 🔮 Future Enhancements
  - Multi-class classification (benign, pre-cancer, cancer subtypes)
  - Advanced XAI with interactive visualizations
  - GAN-based data augmentation for rare cases
  - Edge deployment (mobile apps, local devices)
  - Cloud integration for large-scale hospital screening

## 🏁 Conclusion

- OralScan AI combines accuracy, speed, and explainability to make oral cancer detection:
- Faster ⚡
- More accurate 🎯
- More transparent 🔍
With this tool, clinicians can trust the AI’s decision, see where cancerous regions are, and act early — potentially saving lives ❤️.
