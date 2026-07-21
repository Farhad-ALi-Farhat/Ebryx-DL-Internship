# CNN Image Classification on CIFAR-10

This project demonstrates the complete workflow of building, improving, and evaluating Convolutional Neural Networks (CNNs) for image classification using the **CIFAR-10** dataset. The project progresses from a simple CNN built from scratch to a high-performing transfer learning model using **MobileNetV2**, highlighting the impact of regularization techniques and pretrained models on performance.

---

## Project Objectives

- Build a CNN from scratch for image classification.
- Improve the CNN using Batch Normalization and Dropout.
- Apply Data Augmentation to improve generalization.
- Implement Transfer Learning using MobileNetV2.
- Fine-tune the pretrained model.
- Compare the performance of different approaches.

---

## Dataset

**CIFAR-10**

- **60,000** color images
- **32 × 32** pixels
- **10 classes**

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset Split:

- **Training:** 50,000 images
- **Testing:** 10,000 images

---

## Project Workflow

### 1. Data Preprocessing

- Loaded the CIFAR-10 dataset.
- Normalized images for the custom CNN.
- Preserved raw images for transfer learning to ensure compatibility with MobileNetV2 preprocessing.

---

### 2. Baseline CNN

Built a simple CNN consisting of:

- Convolutional Layers
- Max Pooling Layers
- Flatten Layer
- Dense Layer
- Softmax Output Layer

This model served as the baseline for comparison.

---

### 3. Model Regularization

The baseline CNN was improved by introducing:

- Batch Normalization
- Dropout
- Data Augmentation

These techniques reduced overfitting and improved generalization.

---

### 4. Transfer Learning

Implemented transfer learning using **MobileNetV2** pretrained on ImageNet.

Steps included:

- Loading pretrained weights
- Freezing the feature extraction layers
- Adding a custom classification head
- Training only the classifier

---

### 5. Fine-Tuning

To further improve performance:

- Unfroze the last layers of MobileNetV2
- Reduced the learning rate
- Continued training to adapt pretrained features to CIFAR-10

---

## Model Performance

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|:------|------------------:|--------------------:|--------------:|
| Baseline CNN | **97.6%** | **66.0%** | **65.5%** |
| CNN + Batch Normalization + Dropout + Data Augmentation | **67.0%** | **70.7%** | **70.5%** |
| MobileNetV2 (Frozen) | **86.5%** | **86.1%** | **85.3%** |
| MobileNetV2 (Fine-Tuned) | **92.5%** | **88.9%** | **88.4%** |

---

## Key Observations

- The baseline CNN achieved very high training accuracy but suffered from significant overfitting.
- Batch Normalization, Dropout, and Data Augmentation improved the model's ability to generalize.
- Transfer learning with MobileNetV2 significantly outperformed the custom CNN.
- Fine-tuning the pretrained network further improved classification performance and produced the best overall results.

---

## Debugging & Lessons Learned

During the transfer learning implementation, the model initially achieved only **10–15% accuracy**, which is close to random guessing for a 10-class classification task.

The issue was traced to an input preprocessing mismatch:

- The images had already been normalized to **[0, 1]**.
- `mobilenet_v2.preprocess_input()` expects raw pixel values in **[0, 255]** and performs its own normalization to **[-1, 1]**.

Keeping a separate copy of the raw dataset for transfer learning resolved the issue and improved the model's accuracy to over **85%**.

This highlights the importance of using the preprocessing pipeline expected by pretrained models.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Concepts Covered

- Convolutional Neural Networks (CNNs)
- Convolution & Pooling Layers
- Batch Normalization
- Dropout
- Data Augmentation
- Transfer Learning
- Fine-Tuning
- Image Classification
- Model Evaluation
- Overfitting & Generalization

---

## Future Improvements

- Experiment with EfficientNet or ResNet architectures.
- Perform hyperparameter tuning.
- Add confusion matrix and classification report.
- Visualize feature maps and Grad-CAM explanations.
- Train on a custom image dataset.

---

## Author

**Farhad Ali**
