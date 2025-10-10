# Architecture Project: Gradient Vanishing in MLP

## Overview
This project explores and implements various techniques to mitigate the **vanishing gradient problem** in Multi-Layer Perceptrons (MLPs).  
The baseline MLP model is improved through multiple strategies related to activation functions, optimizers, normalization, and architectural modifications.

## Steps
1. **Prepare Dataset** – Load and preprocess the dataset for training and evaluation.  
2. **Build Baseline MLP** – Implement a simple MLP as the starting point.  
3. **Training** – Train the baseline model and observe vanishing gradient issues.  
4. **Evaluation** – Compare baseline and improved models using metrics and loss curves.  

## Improvements Implemented
- **Weight Initialization** – Improved initialization to stabilize gradient flow.  
- **Activation Functions** – Replace sigmoid/tanh with ReLU and other non-saturating functions.  
- **Optimizers** – Use advanced optimizers (Adam, RMSProp) instead of vanilla SGD.  
- **Normalization** – Apply batch normalization to reduce internal covariate shift.  
- **Skip Connections** – Add residual/shortcut connections to ease gradient propagation.  
- **Layer-wise Training** – Train deeper layers progressively to reduce gradient issues.  
- **Gradient Normalization** – Apply techniques to rescale gradients and prevent vanishing.  

## Tools & Libraries Used
- Python  
- PyTorch  
- NumPy  
- Matplotlib
