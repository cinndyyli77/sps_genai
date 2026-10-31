# Assignment 3: MNIST Image Generation
Author: Xinying (Cindy) Li

## Model
The generator maps 100-dimensional Gaussian noise through a fully
connected layer to 128 x 7 x 7 features. Two transposed convolutions
produce 64 x 14 x 14 features and a 1 x 28 x 28 grayscale image.
Batch normalization and ReLU follow the first transposed convolution;
Tanh is the output activation.

The discriminator uses two stride-2 convolutions with 64 and 128
channels, LeakyReLU activations, batch normalization after the second
convolution, and a fully connected layer with Sigmoid output.

## Training
The GAN was trained on the MNIST training set for 20 epochs.
Images are normalized to [-1, 1]. Training uses binary cross-entropy
and Adam optimizers with learning rate 0.0002 and betas (0.5, 0.999).

To retrain in an environment with the required dependencies:
    python train_mnist_gan.py --epochs 20

Retraining replaces mnist_gan_checkpoint.pth.
The included checkpoint allows inference without retraining.

## Docker
Run these commands from the repository root with Docker running:
    docker build -t assignment3-mnist .
    docker run --rm -p 8001:8000 assignment3-mnist

Open http://localhost:8001/docs

The container includes the trained checkpoint and runs inference on CPU.
Internet access is required during the image build to install dependencies.

## Generate images
POST /generate_mnist
Request body:
    {"num_samples": 16, "seed": 42}

The response is an image/png grid of generated handwritten digits.
num_samples accepts integers from 1 to 64.
Using the same seed and sample count reproduces the result in the same environment.

Command-line example:
    curl --fail -X POST http://localhost:8001/generate_mnist -H "Content-Type: application/json" -d '{"num_samples":16,"seed":42}' --output generated_mnist.png

## Files
- helper_lib/mnist_gan.py: generator and discriminator
- train_mnist_gan.py: MNIST training and checkpoint saving
- app/mnist_api.py: image generation endpoint
- app/main.py: router integration
- mnist_gan_checkpoint.pth: trained model weights
- outputs/mnist_gan/epoch_020.png: final fixed-noise samples
- outputs/mnist_gan/losses.csv: training loss history

## Validation
The architecture check produced generator output (4, 1, 28, 28)
and discriminator output (4, 1).
The Docker API returned HTTP 200 and a PNG grid for 16 samples.
Many generated digits are recognizable; some remain distorted.
