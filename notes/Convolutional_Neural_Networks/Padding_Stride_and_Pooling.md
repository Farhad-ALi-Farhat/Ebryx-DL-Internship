# Padding, Stride, and Pooling

After convolution, three important concepts determine how a CNN processes an image:

1. **Stride** – How far the filter moves each step.
2. **Padding** – Whether extra pixels are added around the image.
3. **Pooling** – How the feature maps are downsampled.

These operations control the output size, computational cost, and the amount of information retained.

---

# 1. Stride

## What is Stride?

**Stride** is the number of pixels the filter moves after each convolution operation.

For example:

- **Stride = 1** → Move one pixel at a time.
- **Stride = 2** → Move two pixels at a time.
- **Stride = 3** → Move three pixels at a time.

A larger stride produces a smaller output feature map because the filter skips more positions.

---

## Example: Stride = 1

Suppose we have a **5 × 5** image and a **3 × 3** filter.

The filter moves one pixel at a time.

```text
Step 1       Step 2

+---+        +---+
|###|        .+---+
|###|  --->  .|###|
|###|        .|###|
+---+        .+---+
```

The filter visits every possible location.

---

## Example: Stride = 2

Now the filter moves two pixels at a time.

```text
+---+      ..+---+
|###| ---> ..|###|
|###|      ..|###|
|###|      ..+---+
+---+
```

Many intermediate positions are skipped.

---

## Output Size Formula

For an input image:

$$
N \times N
$$

Filter size:

$$
F
$$

Padding:

$$
P
$$

Stride:

$$
S
$$

The output size is

$\boxed{\text{Output Size}=\left\lfloor\frac{N-F+2P}{S}\right\rfloor+1}$

where:

- \(N\) = Input size
- \(F\) = Filter size
- \(P\) = Padding
- \(S\) = Stride

---

## Example

Input:

$$
32\times32
$$

Filter:

$$
3\times3
$$

Padding:

$$
0
$$

Stride:

$$
1
$$

Output:

$$
\frac{32-3+0}{1}+1
=30
$$

Result:

$$
30\times30
$$

---

## Another Example

Input:

$$
32\times32
$$

Filter:

$$
3\times3
$$

Stride:

$$
2
$$

Output:


$\left\lfloor\frac{32-3}{2}\right\rfloor+1$
= 15+1
= 16
Final output:

$$
16\times16
$$

Notice how increasing the stride reduces the spatial dimensions.

---

# Why Use Larger Strides?

Advantages:

- Faster computation
- Smaller feature maps
- Less memory usage

Disadvantages:

- May lose fine details
- Small objects can be missed

Most CNNs use:

$$
\boxed{\text{Stride} = 1}
$$

and reduce image size later using pooling.

---

# 2. Padding

## Why Do We Need Padding?

Every convolution without padding reduces the image size.

Example:

Input:

$$
5\times5
$$

Filter:

$$
3\times3
$$

Output:

$$
3\times3
$$

If we keep applying convolutions, the image shrinks rapidly.

Padding solves this problem.

---

## What is Padding?

Padding means adding extra pixels around the border of the image.

Usually, these added pixels are zeros, so it is often called **zero padding**.

Example:

Original image:

```text
1 2 3
4 5 6
7 8 9
```

After adding one layer of zero padding:

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

---

## Benefits of Padding

### 1. Preserve Image Size

Without padding:

$$
32\times32
\rightarrow
30\times30
$$

With padding:

$$
32\times32
\rightarrow
32\times32
$$

(if using a 3 × 3 filter with stride = 1)

---

### 2. Preserve Border Information

Without padding, pixels on the edges are used fewer times than pixels in the center.

Padding allows edge pixels to contribute more equally during convolution.

---

## Types of Padding

### Valid Padding

No padding is added.

$$
P=0
$$

Output becomes smaller.

---

### Same Padding

Padding is added so that the output size remains the same as the input (for stride = 1).

Example:

Input:

$$
64\times64
$$

Output:

$$
64\times64
$$

TensorFlow:

```python
padding="same"
```

---

## Example

Input:

$$
28\times28
$$

Filter:

$$
3\times3
$$

Stride:

$$
1
$$

Padding:

$$
1
$$

Output:

$\frac{28-3+2(1)}{1}+1=28$

So the output remains

$$
28\times28
$$

---

# 3. Pooling

## What is Pooling?

Pooling reduces the size of feature maps while keeping the most important information.

Instead of learning parameters, pooling performs a fixed mathematical operation.

Benefits:

- Reduces computation
- Reduces memory usage
- Helps prevent overfitting
- Makes the network more robust to small shifts in the input

---

# Max Pooling

The most commonly used pooling operation.

A small window (typically **2 × 2**) slides across the feature map and selects the maximum value.

Example:

Input:

$$
\begin{bmatrix}
1 & 3\\
4 & 2
\end{bmatrix}
$$

Maximum value:

$$
4
$$

Output:

$$
4
$$

---

## Larger Example

Input:

$$
\begin{bmatrix}
1 & 2 & 3 & 0\\
4 & 6 & 5 & 1\\
7 & 2 & 9 & 3\\
4 & 8 & 1 & 5
\end{bmatrix}
$$

Using a **2 × 2** pooling window with **stride = 2**:

Window 1:

$$
\begin{bmatrix}
1 & 2\\
4 & 6
\end{bmatrix}
\rightarrow
6
$$

Window 2:

$$
\begin{bmatrix}
3 & 0\\
5 & 1
\end{bmatrix}
\rightarrow
5
$$

Window 3:

$$
\begin{bmatrix}
7 & 2\\
4 & 8
\end{bmatrix}
\rightarrow
8
$$

Window 4:

$$
\begin{bmatrix}
9 & 3\\
1 & 5
\end{bmatrix}
\rightarrow
9
$$

Final output:

$$
\begin{bmatrix}
6 & 5\\
8 & 9
\end{bmatrix}
$$

The spatial dimensions are reduced from **4 × 4** to **2 × 2**.

---

# Average Pooling

Instead of taking the maximum value, average pooling computes the mean of all values in the pooling window.

Example:

Input:

$$
\begin{bmatrix}
2 & 4\\
6 & 8
\end{bmatrix}
$$

Output:

$$
\frac{2+4+6+8}{4}=5
$$

Average pooling is less common in modern CNNs because max pooling tends to preserve the most distinctive features.

---

# Typical Pooling Parameters

Most CNN architectures use:

- Pool size: **2 × 2**
- Stride: **2**

This reduces the height and width by half.

Example:

$$
32\times32
\rightarrow
16\times16
$$

---

# Convolution vs Pooling

| Convolution | Pooling |
|------------|---------|
| Learns filters during training | No learnable parameters |
| Extracts features | Reduces feature map size |
| Detects edges, textures, shapes | Keeps the most important information |
| Produces feature maps | Produces smaller feature maps |

---

# Typical CNN Block

A CNN often stacks these operations in the following order:

```text
Input Image
      │
      ▼
Convolution
      │
      ▼
ReLU
      │
      ▼
Max Pooling
      │
      ▼
Convolution
      │
      ▼
ReLU
      │
      ▼
Max Pooling
      │
      ▼
Flatten
      │
      ▼
Dense Layer
      │
      ▼
Output
```

---

# Key Takeaways

- **Stride** determines how far the filter moves across the image.
- A larger stride produces smaller feature maps but may lose fine details.
- **Padding** adds extra pixels (usually zeros) around the image to preserve spatial dimensions and retain border information.
- **Pooling** reduces the size of feature maps, making the network faster and helping reduce overfitting.
- **Max pooling** is the most widely used pooling operation because it preserves the strongest feature responses.
- A typical CNN repeatedly applies **Convolution → ReLU → Pooling** before passing the extracted features to fully connected layers.
