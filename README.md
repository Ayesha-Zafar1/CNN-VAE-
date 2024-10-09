# CNN Variational Autoencoder (VAE)

This project implements a Convolutional Variational Autoencoder (CNN-VAE) using PyTorch to reconstruct images from the MNIST dataset.

# Overview

The model is designed to encode MNIST images into a latent space and decode them back to reconstruct the original images. It uses convolutional layers for both the encoder and decoder, and the loss function combines reconstruction loss (Binary Cross Entropy) and KL divergence to regularize the latent space.

# Requirements

PyTorch
torchvision
matplotlib
tqdm
numpy

# Model Structure

The VAE consists of:

- Encoder: Convolutional layers to downsample images.
- Latent Space: Mean (mu) and log-variance (logvar) computed from encoded features.
- Decoder: Transposed convolutional layers to reconstruct the images.

# Training and Evaluation

The model is trained with a combination of reconstruction loss and KL divergence. Progress is updated after each batch during training. Evaluation involves visualizing reconstructed images and plotting the latent space.

# Hyperparameters

I experiment with the following hyperparameters:

- Batch Size: 64, 32, 128
- Learning Rate:1e-4, 5e-4, 1e-3
- Epochs: 10, 20, 30, 50
- Latent Dim: 2

# Visualizations

The project includes functions to visualize:

- Reconstructed images from the latent space.
- Latent space clustering of MNIST digits.

# How to Run

- Clone the repository and install the required dependencies.
- Train the model using the provided training script.
- Visualize the results, including image reconstructions and latent space embeddings.
