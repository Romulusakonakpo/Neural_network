# Neural Network for Handwritten Digit Classification (MNIST-like)

## Project Overview  
This project implements a deep neural network from scratch to classify handwritten digits from the MNIST-like `digits` dataset (8x8 images). It demonstrates key deep learning techniques including multiple hidden layers, dropout, batch normalization, mini-batch training, and optimization improvements.

## Key Objectives  
- Build a fully connected neural network with two hidden layers  
- Apply dropout regularization and batch normalization  
- Train the model using mini-batch gradient descent  
- Implement cross-entropy loss with momentum optimizer  
- Incorporate learning rate decay and early stopping for robust training  
- Evaluate model performance with accuracy, confusion matrix, and error analysis  

## Dataset  
- `sklearn.datasets.load_digits` — 1797 samples, 8x8 grayscale images of digits 0-9  
- Data normalized between 0 and 1  
- Train/test split: 80% training, 20% testing  

## Methodology  
1. Initialize weights using He initialization  
2. Forward propagation with ReLU activations and dropout  
3. Batch normalization after each hidden layer  
4. Compute cross-entropy loss  
5. Backpropagation with momentum optimizer  
6. Early stopping to prevent overfitting  
7. Learning rate decay for training stability  
8. Final evaluation using accuracy and confusion matrix  

## Results  
- Achieved accuracy of ~97% on test set  
- Detailed classification report for each digit class  
- Confusion matrix visualization highlighting common misclassifications  
- Analysis and visualization of misclassified digits  

## Tools & Technologies  
- Python with NumPy and Matplotlib  
- Scikit-learn for data loading and evaluation metrics  
- Custom implementation of neural network components (no deep learning frameworks)  

---

Feel free to explore the notebooks and scripts for full implementation details and training results.
