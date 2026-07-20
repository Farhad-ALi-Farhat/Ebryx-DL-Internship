# CNN Architectures: LeNet, VGG, and ResNet (Introduction)

Over the years, Convolutional Neural Networks (CNNs) have evolved from simple networks with a few layers to extremely deep architectures capable of solving complex computer vision tasks.

Some of the most influential CNN architectures are:

1. **LeNet-5 (1998)** – The first successful CNN for handwritten digit recognition.
2. **VGG (2014)** – Demonstrated that deeper networks with small filters significantly improve performance.
3. **ResNet (2015)** – Introduced residual connections, enabling very deep networks to train effectively.

These architectures laid the foundation for many modern computer vision models.

---

# Evolution of CNN Architectures

```text
LeNet (1998)
      │
      ▼
AlexNet (2012)
      │
      ▼
VGG (2014)
      │
      ▼
GoogLeNet (2014)
      │
      ▼
ResNet (2015)
      │
      ▼
DenseNet, EfficientNet, ConvNeXt, Vision Transformers...
```

Each new architecture addressed limitations of earlier models, such as limited depth, computational cost, or optimization difficulties.

---

# 1. LeNet-5

## Introduction

LeNet-5 was proposed by **Yann LeCun** and colleagues in **1998**. It is considered one of the first practical CNNs and was designed to recognize handwritten digits.

It was trained on the **MNIST** dataset, which contains grayscale images of handwritten digits (0–9).

---

## Input

Image size:

$$
32 \times 32 \times 1
$$

Although MNIST images are **28 × 28**, they were padded to **32 × 32** in the original LeNet-5 implementation.

---

## Architecture

```text
Input (32×32×1)
        │
        ▼
Convolution
        │
        ▼
Average Pooling
        │
        ▼
Convolution
        │
        ▼
Average Pooling
        │
        ▼
Flatten
        │
        ▼
Fully Connected
        │
        ▼
Fully Connected
        │
        ▼
Output (10 classes)
```

---

## Characteristics

- Used **5 × 5** convolution filters.
- Used **Average Pooling** instead of Max Pooling.
- Contained only a few layers.
- Designed for grayscale images.

---

## Advantages

- Simple architecture
- Easy to understand
- Low computational cost
- Excellent for educational purposes

---

## Limitations

- Very shallow compared to modern networks
- Cannot effectively learn highly complex visual features
- Not suitable for large-scale image recognition tasks

---

# 2. VGG Network

## Introduction

The **VGG** network was introduced by researchers at the **Visual Geometry Group (VGG)**, University of Oxford, in **2014**.

The key insight was surprisingly simple:

> Instead of using large convolution filters (e.g., 7 × 7 or 11 × 11), stack many **3 × 3** convolution layers.

This made the network deeper while keeping the number of parameters manageable.

---

## Popular Variants

The number in the name refers to the total number of learnable layers.

| Model | Learnable Layers |
|--------|------------------|
| VGG11 | 11 |
| VGG13 | 13 |
| VGG16 | 16 |
| VGG19 | 19 |

Among these, **VGG16** is the most widely used.

---

## VGG16 Architecture

```text
Input
  │
  ▼
Conv
Conv
Pool
  │
  ▼
Conv
Conv
Pool
  │
  ▼
Conv
Conv
Conv
Pool
  │
  ▼
Conv
Conv
Conv
Pool
  │
  ▼
Conv
Conv
Conv
Pool
  │
  ▼
Flatten
  │
  ▼
Dense
Dense
Output
```

---

## Key Characteristics

### Small Filters

Every convolution uses:

$$
3 \times 3
$$

This became a standard design choice for many later CNNs.

---

### Max Pooling

Pooling layer:

$$
2 \times 2
$$

Stride:

$$
2
$$

This reduces the height and width by half after each pooling operation.

---

### Deep Network

Compared to LeNet, VGG contains many more convolution layers.

This allows it to learn:

- Edges
- Textures
- Object parts
- Entire objects

---

## Advantages

- Excellent feature extraction
- Simple and uniform architecture
- Easy to understand and implement
- Widely used for transfer learning

---

## Limitations

- Very large number of parameters (about **138 million** in VGG16).
- High memory usage
- Slow training and inference compared to more modern architectures

---

# 3. ResNet (Residual Network)

## Introduction

As researchers built deeper CNNs, they noticed a surprising problem:

> After a certain depth, adding more layers sometimes **reduced** accuracy instead of improving it.

This was not just due to overfitting—it became harder to optimize very deep networks because gradients could diminish or explode as they propagated backward.

To address this, **ResNet** introduced **residual (skip) connections**, making it much easier to train deep networks.

---

## The Main Idea

Instead of learning

$$
H(x)
$$

ResNet learns the **residual**

$$
F(x)=H(x)-x
$$

The original input is then added back:

$$
H(x)=F(x)+x
$$

This shortcut makes it easier for the network to preserve useful information and optimize very deep models.

---

## Residual Block

Traditional CNN:

```text
Input
  │
  ▼
Conv
  │
  ▼
Conv
  │
  ▼
Output
```

ResNet:

```text
           ┌──────────────┐
           │              │
Input ───► Conv ─► Conv ─► + ─► Output
   │                       ▲
   └───────────────────────┘
```

The shortcut connection bypasses the convolution layers and is added to their output.

---

## Popular Variants

| Model | Approximate Layers |
|--------|--------------------|
| ResNet18 | 18 |
| ResNet34 | 34 |
| ResNet50 | 50 |
| ResNet101 | 101 |
| ResNet152 | 152 |

Despite their depth, these models can be trained effectively because of residual connections.

---

## Why Skip Connections Help

Without skip connections:

```text
Input
  │
Conv
  │
Conv
  │
Conv
  │
Output
```

Information and gradients must pass through every layer.

With skip connections:

```text
Input ─────────────────────┐
  │                        │
Conv                        │
  │                        │
Conv                        │
  │                        │
  └──────────────► Add ◄────┘
                   │
                   ▼
                 Output
```

The shortcut provides an additional path for information and gradients, making optimization easier.

---

## Advantages

- Enables very deep networks
- Easier optimization
- Better accuracy on many vision tasks
- Reduced vanishing gradient problem
- Widely used in image classification, object detection, and segmentation

---

## Limitations

- More complex than LeNet and VGG
- Higher computational requirements for deeper variants
- Less intuitive architecture because of shortcut connections

---

# Architecture Comparison

| Feature | LeNet | VGG | ResNet |
|----------|--------|------|---------|
| Year | 1998 | 2014 | 2015 |
| Typical Depth | 5–7 layers | 11–19 layers | 18–152+ layers |
| Main Idea | First practical CNN | Deep network with small 3 × 3 filters | Residual (skip) connections |
| Pooling | Average Pooling | Max Pooling | Max Pooling |
| Suitable For | Simple image classification | General image classification | Large-scale and deep computer vision tasks |
| Computational Cost | Low | High | Moderate to High (depending on depth) |

---

# Choosing an Architecture

### LeNet

Use when:

- Learning CNN fundamentals
- Working with small grayscale datasets like MNIST
- Computational resources are limited

---

### VGG

Use when:

- High-quality feature extraction is needed
- Transfer learning on moderate-sized datasets
- Simplicity is preferred over efficiency

---

### ResNet

Use when:

- Solving complex computer vision tasks
- Training or fine-tuning deep networks
- High accuracy is important

ResNet is one of the most commonly used backbone architectures in modern computer vision.

---

# Summary Timeline

```text
1998
LeNet
│
├── First successful CNN
├── Handwritten digit recognition
└── Average Pooling

        ↓

2014
VGG
│
├── Deep architecture
├── 3 × 3 convolutions
└── Max Pooling

        ↓

2015
ResNet
│
├── Residual (skip) connections
├── Enables very deep networks
└── Foundation for many modern CNN models
```

---

# Key Takeaways

- **LeNet** was the first successful CNN and demonstrated the effectiveness of convolutional layers for image recognition.
- **VGG** showed that deeper networks built from small **3 × 3** filters can achieve excellent performance and became a popular choice for transfer learning.
- **ResNet** introduced residual (skip) connections, allowing very deep CNNs to be trained efficiently by improving gradient flow.
- Modern computer vision models build upon these ideas, making LeNet, VGG, and ResNet essential architectures to understand before exploring advanced CNNs.
