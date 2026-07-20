# Convolution, Filters, and Feature Maps

## What is Convolution?

Convolution is the fundamental operation in a Convolutional Neural Network (CNN). Instead of connecting every input pixel to every neuron (as in an ANN), a CNN applies a **small matrix called a filter (or kernel)** that slides across the image to detect useful patterns.

These patterns can include:

- Edges
- Corners
- Textures
- Shapes
- Eyes
- Wheels
- Faces

Unlike traditional image processing where filters are manually designed, CNNs **learn the filter values automatically during training**.

---

# Why Convolution?

Consider an RGB image of size:

$$
224 \times 224 \times 3
$$

A fully connected layer with 100 neurons would require:

$$
224 \times 224 \times 3 \times 100
=
15,052,800
$$

weights.

This is computationally expensive and ignores the spatial relationships between nearby pixels.

Convolution solves this by:

- Using small filters
- Sharing the same filter across the entire image
- Learning local patterns first

---

# What is a Filter (Kernel)?

A **filter** (also called a **kernel**) is a small matrix that scans over the input image.

Example:

$$
\begin{bmatrix}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
$$

A filter is much smaller than the image.

Typical filter sizes are:

- 3 × 3
- 5 × 5
- 7 × 7

Modern CNNs usually use **3 × 3 filters** because they capture local patterns efficiently while keeping the number of parameters low.

---

# How Convolution Works

Suppose our input image is

$$
\begin{bmatrix}
1 & 2 & 0 & 1\\
4 & 3 & 1 & 2\\
2 & 1 & 0 & 1\\
1 & 2 & 3 & 4
\end{bmatrix}
$$

and our filter is

$$
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
$$

### Step 1

Place the filter over the first 2 × 2 region:

$$
\begin{bmatrix}
1 & 2\\
4 & 3
\end{bmatrix}
$$

Multiply element-wise:

$$
(1\times1)+(2\times0)+(4\times0)+(3\times1)
$$

$$
=1+0+0+3=4
$$

The first value of the output becomes:

$$
4
$$

---

### Step 2

Move the filter one step to the right.

Image patch:

$$
\begin{bmatrix}
2 & 0\\
3 & 1
\end{bmatrix}
$$

Compute:

$$
(2\times1)+(0\times0)+(3\times0)+(1\times1)
$$

$$
=3
$$

Continue until the filter has scanned the whole image.

---

# Mathematical Formula

For an input image \(I\) and filter \(K\), the convolution output is

$$
S(i,j)
=
\sum_m
\sum_n
I(i+m,j+n)
K(m,n)
$$

where:

- \(I\) is the input image
- \(K\) is the filter
- \(S\) is the output feature map

---

# What Does the Filter Learn?

Different filters detect different visual patterns.

Example:

Vertical edges

$$
\begin{bmatrix}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
$$

Horizontal edges

$$
\begin{bmatrix}
1 & 1 & 1\\
0 & 0 & 0\\
-1 & -1 & -1
\end{bmatrix}
$$

Blur

$$
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{bmatrix}
$$

Sharpen

$$
\begin{bmatrix}
0 & -1 & 0\\
-1 & 5 & -1\\
0 & -1 & 0
\end{bmatrix}
$$

In image processing these filters are designed manually.

In CNNs, these values are **learned automatically** using backpropagation.

---

# What is a Feature Map?

The output produced after applying a filter is called a **feature map** (also known as an activation map).

Suppose the input image is

$$
5\times5
$$

and we use a

$$
3\times3
$$

filter.

The output becomes

$$
3\times3
$$

(assuming stride = 1 and no padding).

Each value in the feature map represents **how strongly that region of the image matches the filter**.

Higher values indicate a stronger presence of the learned feature.

---

# Multiple Filters

CNNs do **not** use just one filter.

Instead, they learn many filters simultaneously.

Example:

- Filter 1 detects vertical edges.
- Filter 2 detects horizontal edges.
- Filter 3 detects curves.
- Filter 4 detects textures.
- Filter 5 detects corners.

If we use 32 filters, we get:

- 32 feature maps

If we use 64 filters:

- 64 feature maps

The number of filters determines the **depth** (number of channels) of the output.

---

# Example

Input image:

$$
32\times32\times3
$$

Apply:

- 32 filters
- Each filter size:

$$
3\times3
$$

Output:

$$
30\times30\times32
$$

Each of the 32 filters produces one feature map.

---

# Why Are Multiple Filters Important?

A single filter can only detect one type of pattern.

Real-world images contain many different patterns simultaneously.

Using multiple filters allows the CNN to learn:

- Edges
- Colors
- Shapes
- Curves
- Textures
- Object parts

These simple features are combined in deeper layers to recognize complex objects such as faces, animals, or vehicles.

---

# How CNN Learns Filters

Initially, filter values are random.

Example:

$$
\begin{bmatrix}
0.14 & -0.82 & 0.31\\
0.77 & -0.41 & 0.18\\
-0.29 & 0.56 & 0.93
\end{bmatrix}
$$

After each training iteration:

1. Forward pass computes predictions.
2. Loss is calculated.
3. Backpropagation computes gradients.
4. The optimizer updates the filter values.

Eventually, the filters evolve to detect meaningful visual patterns automatically.

---

# Key Takeaways

- Convolution is the core operation of a CNN.
- A filter (kernel) is a small matrix that scans over an image.
- Filters extract useful visual features such as edges and textures.
- The output of a filter is called a **feature map**.
- CNNs learn filter values automatically during training.
- Using multiple filters allows a CNN to detect many different patterns at the same time.
