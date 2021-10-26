# Deep Neural Kernel for Gaussian Process Regression Classification
A set of Python tools to for parametrizing the kernel in a Gaussian Process by neural networks.

### Usage
There are two vital files included in this repo 
- DeepKernel.py contains the kernel parametrized by neural networks
- Train.py contains the code to train the model via MLE. Note that the model in Train.py outputs both a case count prediction and a hotspot (binary) prediction

Neural Kernel implementation from https://arxiv.org/pdf/2106.00072.pdf
