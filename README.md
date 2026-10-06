# Deep Learning and Neural Networks with Applications📊

This repository contains my learning materials and lab exercises for Deep Learning and Neural Networks with Applications.


## About Deep Learning and Neural Networks with Applications ❓

This module covers the foundations and modern applications of deep learning, from the building blocks of neural networks to the latest advances in AI reasoning and alignment. The labs progress from implementing a perceptron from scratch, through multi-layer networks, CNNs, transfer learning, RNNs and Transformers, to working with large language models for reasoning tasks.

For practical implementation, Python and PyTorch are used in Jupyter notebooks on Google Colab, with Hugging Face libraries for modern NLP and vision-language model work.


## Environment 👩🏻‍💻

<p align="center">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/>
</p>


## Stack 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/torchvision-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="torchvision"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat&logo=transformers&logoColor=black" alt="Transformers"/>
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="Datasets"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white" alt="matplotlib"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/seaborn-3776AB?style=flat&logo=seaborn&logoColor=white" alt="seaborn"/>
  <img src="https://img.shields.io/badge/Pillow-9C3B1D?style=flat&logo=python&logoColor=white" alt="Pillow"/>
  <img src="https://img.shields.io/badge/yfinance-1F77B4?style=flat&logo=yahoo&logoColor=white" alt="yfinance"/>
  <img src="https://img.shields.io/badge/bitsandbytes-76B900?style=flat&logo=nvidia&logoColor=white" alt="bitsandbytes"/>
</p>


## Overview of Lab Materials 🔬

### Lab 1
**Introduction to Neural Networks**
- Building a perceptron from scratch in pure Python
- Training on logic gates (AND, OR, NAND, XOR)
- Visualising decision boundaries with Matplotlib
- Exploring the effect of training epochs and bias on predictions
- Understanding the limitation of a single perceptron on XOR (non-linear separability)
- Introduction to PyTorch tensors: creating scalars, vectors, matrices and higher-rank tensors
- Tensor data types (int, float, bool) and creation methods (zeros, ones, rand)
- Tensor operations: scalar multiplication, addition, matrix multiplication (matmul)
- Reshaping tensors: flatten, unsqueeze, squeeze and using -1 for inferred dimensions

### Lab 2
**Multiple Layer Neural Networks**
- Loading the Iris dataset and preparing tensors for three-class classification
- Building a multi-layer neural network from scratch with weight initialisation
- Forward pass: hidden layer with sigmoid activation, output layer producing logits
- Multi-class cross-entropy loss with `nn.CrossEntropyLoss`
- Training loop: `zero_grad`, forward pass, loss computation, `backward`, `optimizer.step`
- Converting logits to probabilities with softmax and predictions with argmax
- Rebuilding the model using `nn.Module` for cleaner, modular code
- Comparing manual implementation with the `nn.Module` approach
- Optional: experimenting with hidden size, Adam vs SGD, mini-batch training

### Lab 3
**Training Methods and Optimisation**
- Building a binary classification dataset with `make_moons` from scikit-learn
- Creating a tiny training set (30 samples) to deliberately expose overfitting
- Implementing an oversized MLP with 500 hidden units and ReLU activation
- Training with `BCEWithLogitsLoss` and interpreting accuracy curves
- Observing the gap between training and validation performance (overfitting)
- Applying dropout regularisation and comparing different dropout rates (0.0–0.6)
- Applying weight decay (L2 regularisation) and measuring its effect on generalisation
- Comparing dropout vs weight decay: which improves generalisation more
- Plotting training/validation loss and accuracy curves for baseline vs regularised models

### Lab 4
**Convolutional Neural Networks**
- Understanding image tensor format: channels, height, width
- Implementing a convolutional layer from scratch
- Implementing pooling and flatten operations
- Building a CNN architecture for image classification
- Writing the training loop for a CNN
- Evaluating model performance with a confusion matrix
- Optional: learning rate scheduling, adding dropout, testing on real-world images

### Lab 5
**Pretrained Models and Transfer Learning in Computer Vision**
- Loading a pretrained ResNet backbone and inspecting its feature extraction layers
- Replacing the final classification layer for a new class count
- Freezing early layers with `requires_grad` and training only the new head
- Understanding feature hierarchy: edges/textures, parts/motifs, task-specific layers
- Fine-tuning: swapping the classifier head, strategic freezing, training loop
- Avoiding common mistakes (mixing train/val batches, calling backward during validation)
- Comparing transfer learning vs training from scratch
- Optional: two-stage fine-tuning (train head, then unfreeze all with smaller learning rate)

### Lab 6
**Recurrent Neural Networks**
- Implementing a vanilla RNN for stock price prediction using yfinance data
- Building an LSTM for sentiment analysis on text data
- Text preprocessing: converting words to sequences of numbers using a vocabulary
- Implementing an encoder-decoder LSTM architecture on a synthetic sequence-to-sequence task
- Understanding binary cross-entropy loss for sentiment classification

### Lab 7
**Transformers**
- Coding scaled dot-product attention manually with Q, K, V matrices
- Defining and training a tiny decoder-only Transformer from scratch for character-level next-token prediction
- Sentiment analysis with encoder-only Transformers using BERT
- Using a pre-trained BERT model for a simple classification task
- Learning to fine-tune a pre-trained model on a custom dataset

### Lab 8
**Applications of RNNs and Transformers**
- Comparing discriminative vs generative approaches for a classification task on a real-world dataset
- Using a small Vision-Language Model (VLM) for multimodal visual question answering
- Exploring how generative models can support classification through prompting
- Understanding the trade-offs between discriminative fine-tuning and generative approaches

### Lab 9
**Modern Neural Networks: AI Reasoning**
- Zero-shot Chain-of-Thought (CoT) prompting to solve reasoning problems
- Self-Consistency (SC) prompting: sampling multiple reasoning paths and aggregating
- Implementing reasoning tasks with an LLM pre-trained for reasoning (native reasoning model)
- Comparing prompting strategies and their effect on reasoning accuracy


## Repository Structure 🌲

```
.
├──.gitattributes
├── W1 - Lab1
│   └── Lab_1.ipynb
├── W2 - Lab2
│   └── Lab_2.ipynb
├── W3 - Lab3
│   └── Lab_3.ipynb
├── W4 - Lab4
│   └── Lab_4.ipynb
├── W5 - Lab5
│   └── Lab_5.ipynb
├── W6 - Lab6
│   └── Lab_6.ipynb
├── W7 - Lab7
│   └── Lab_7.ipynb
├── W8 - Lab8
│   └── Lab_8.ipynb
└── W9 - Lab9
    └── Lab_9.ipynb
├── README.md
```


## Reflection 🪞

Across these labs I built up the foundations of deep learning step by step. I started with a perceptron from scratch, learning how weights, bias and decision boundaries work, and why a single perceptron cannot solve XOR. Moving to PyTorch tensors gave me the tools to build multi-layer networks, and in Lab 2 I implemented an MLP on the Iris dataset, learning the full training workflow from forward pass to backpropagation and parameter updates.

Lab 3 was a turning point: working with a deliberately tiny training set, I saw overfitting happen in real time and experimented with dropout and weight decay to improve generalisation. Building CNNs in Lab 4 taught me how convolutional layers extract spatial features from images, and transfer learning in Lab 5 showed me how pretrained models like ResNet can be adapted to new tasks with minimal training.

The RNN labs introduced me to sequential data, from stock price prediction to sentiment analysis with LSTMs and encoder-decoder architectures. Transformers were the biggest leap: I implemented attention from scratch, trained a decoder-only transformer, and fine-tuned BERT for sentiment analysis. The final labs on VLMs and LLM reasoning opened my eyes to how modern AI systems can be applied through prompting, chain-of-thought reasoning and multimodal understanding.

Looking ahead, I want to explore larger-scale model training, experiment with more advanced architectures, and engage with the ethical considerations around fairness, accountability and transparency in deep learning systems.
