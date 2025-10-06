# CNN Architecture Implementation

## Learning Objectives

By completing this assignment, you will:

1. **Implement CNN layers from scratch** - Build Conv2D, MaxPool2D, and other essential layers
2. **Understand convolution operations** - Master the mathematics behind convolution and its backpropagation
3. **Design CNN architectures** - Implement LeNet-5 and a simplified VGG network
4. **Apply CNNs to image classification** - Train on CIFAR-10 dataset
5. **Visualize CNN features** - Understand what CNNs learn through filter and feature map visualization
6. **Explore transfer learning** - Use pre-trained features for improved performance

## Assignment Overview

In this assignment, you will build Convolutional Neural Networks (CNNs) from the ground up. Starting with individual layer implementations, you'll progress to constructing complete architectures like LeNet-5 and a mini version of VGG. You'll then apply these networks to the CIFAR-10 image classification task, visualize what your networks learn, and explore transfer learning techniques.

## Part 1: CNN Layer Implementation (40 points)

### 1.1 Conv2D Layer (20 points)
Implement a 2D convolution layer with:
- Forward pass using convolution operation
- Backward pass computing gradients w.r.t. input, weights, and bias
- Support for different padding modes ('valid', 'same')
- Stride support
- Efficient implementation using vectorized operations

### 1.2 MaxPool2D Layer (10 points)
Implement max pooling with:
- Forward pass tracking max indices
- Backward pass routing gradients correctly
- Support for different pool sizes and strides

### 1.3 Additional Layers (10 points)
- Flatten layer for transitioning from conv to fully connected
- Batch normalization for CNNs
- Dropout2D for spatial dropout

## Part 2: CNN Architectures (30 points)

### 2.1 LeNet-5 Implementation (15 points)
Build the classic LeNet-5 architecture:
```
Input (32x32x3) → Conv(6, 5x5) → ReLU → MaxPool(2x2) → 
Conv(16, 5x5) → ReLU → MaxPool(2x2) → 
Flatten → FC(120) → ReLU → FC(84) → ReLU → FC(10)
```

### 2.2 Mini-VGG Implementation (15 points)
Build a simplified VGG-style network:
```
Input (32x32x3) → 
Conv(32, 3x3) → ReLU → Conv(32, 3x3) → ReLU → MaxPool(2x2) →
Conv(64, 3x3) → ReLU → Conv(64, 3x3) → ReLU → MaxPool(2x2) →
Conv(128, 3x3) → ReLU → Conv(128, 3x3) → ReLU → MaxPool(2x2) →
Flatten → FC(256) → ReLU → Dropout(0.5) → FC(10)
```

## Part 3: CIFAR-10 Classification (20 points)

### 3.1 Data Preprocessing (5 points)
- Implement data normalization
- Create data augmentation pipeline (rotation, flipping, cropping)
- Split data appropriately

### 3.2 Training (10 points)
- Train both LeNet-5 and Mini-VGG on CIFAR-10
- Implement proper training loop with validation
- Use appropriate learning rate scheduling
- Track and plot training/validation metrics

### 3.3 Evaluation (5 points)
- Achieve at least 70% accuracy on test set with LeNet-5
- Achieve at least 80% accuracy on test set with Mini-VGG
- Generate confusion matrix and per-class accuracy

## Part 4: Feature Visualization (10 points)

### 4.1 Filter Visualization (5 points)
- Visualize learned filters in first convolutional layer
- Create grid visualization of all filters
- Analyze patterns learned by filters

### 4.2 Feature Map Visualization (5 points)
- Visualize intermediate feature maps for sample images
- Show how features become more abstract in deeper layers
- Implement guided backpropagation or similar technique

## Bonus: Transfer Learning (10 points)

### Implement transfer learning:
- Load pre-trained features (provided)
- Fine-tune on CIFAR-10 with frozen early layers
- Compare performance with training from scratch
- Analyze which layers benefit most from pre-training

## Implementation Requirements

### Code Structure
Your implementation should include:
- `cnn_layers.py`: All layer implementations
- `architectures.py`: LeNet-5 and Mini-VGG classes
- `train.py`: Training script with data loading and augmentation
- `visualize.py`: Visualization utilities
- `transfer_learning.py`: Transfer learning implementation

### Performance Requirements
- Conv2D should use efficient vectorized operations
- Training should utilize GPU if available
- Code should be well-documented with docstrings
- Include timing benchmarks for forward/backward passes



## Getting Started

1. Review the provided starter code in `starter_code.py`
2. Start with implementing basic layers (Conv2D, MaxPool2D)
3. Test your implementations with the provided unit tests
4. Build the architectures once layers are working
5. Train on CIFAR-10 and tune hyperparameters
6. Create visualizations to understand what your network learns

## Resources

- [CS231n CNN Notes](http://cs231n.github.io/convolutional-networks/)
- [Understanding Convolutions](https://github.com/vdumoulin/conv_arithmetic)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Original papers: [LeNet-5](http://yann.lecun.com/exdb/lenet/), [VGG](https://arxiv.org/abs/1409.1556)
