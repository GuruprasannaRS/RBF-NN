# Radial Basis Function Neural Network (RBF-NN) Implementation

## Overview
This project implements a Radial Basis Function Neural Network with Gaussian kernel function and constant spread function. The implementation uses all points in the input as centers for the Radial Basis Functions.

## Architecture
The RBF Neural Network consists of three layers:
- **Input Layer**
- **Hidden Layer** (only one hidden layer, unlike MLP)
- **Output Layer**

Key characteristics:
- Weights between inputs and hidden layer are unity
- Uses Gaussian kernel functions
- Implements pseudo-inverse method for weight calculation

## Dataset Generation
The input samples are created based on the following conditions:
- x1 = -2 + 0.2*xi (where xi is randomly sampled from 0 to 20)
- x2 = -2 + 0.2*xj (where xj is randomly sampled from 0 to 20)
- Output is 1 if (x1² + x2²) ≤ 1, otherwise -1
- Total of 441 samples generated

## Data Split
- **Training Data**: 80% (352 samples)
- **Testing Data**: 20% (89 samples)

## Implementation Details

### Core Functions
1. **Distance Function**: Calculates Euclidean distance between inputs and centers
2. **Gaussian Matrix Computation**: Computes Gaussian kernel matrix using centers and sigma
3. **Weight Matrix Calculation**: Uses pseudo-inverse method to calculate weights
4. **Output Matrix Calculation**: Computes final output using Gaussian matrix and weights
5. **Mean Square Error**: Calculates MSE for performance evaluation
6. **Accuracy Calculation**: Computes classification accuracy

### Three Center Selection Methods

#### 1. All Points as Centers
- Uses all training input points as RBF centers
- Tests sigma values from 0.1 to 10

#### 2. Random Selection (150 centers)
- Randomly selects 150 centers from input data
- Maintains same testing procedure

#### 3. K-Means Clustering (150 centers)
- Uses K-means algorithm to find 150 optimal centers
- Implements scikit-learn KMeans with random_state=0

## Results and Analysis

### Performance Metrics
The implementation evaluates:
- Mean Squared Error (MSE)
- Training Accuracy
- Testing Accuracy

### Key Findings
- **Optimal Sigma Range**: Sigma values between 0.1-0.3 achieve highest accuracy (~99%)
- **Stability**: After sigma=7, accuracy becomes constant for all methods
- **Overfitting**: Small sigma values (0.1-0.3) show high training accuracy but may overfit
- **Generalization**: Mid-range sigma values (5-7) provide balanced performance

### Best Performance
- **All Centers Method**: 98.876% test accuracy with sigma=0.1-0.3
- **Random 150 Centers**: 98.876% test accuracy with sigma=0.3
- **K-Means Centers**: 98.876% test accuracy with sigma=0.2-0.3, 0.7

## Dependencies
- numpy
- matplotlib.pyplot
- sklearn.model_selection (train_test_split)
- sklearn.cluster (KMeans)

## Usage
The notebook contains complete implementation with:
- Data generation and preprocessing
- Model training and testing
- Performance evaluation across different sigma values
- Visualization of results through plots
- Comparison of different center selection methods

## Files
- `RBF_NN.ipynb`: Main implementation notebook
- `README.md`: This documentation file