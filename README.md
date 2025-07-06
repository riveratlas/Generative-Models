# 🎨 State-of-the-Art Generative Models

A curated collection of cutting-edge generative models for various data modalities including images, text, audio, and more. This repository serves as a comprehensive reference for researchers, developers, and enthusiasts in the field of generative AI.

## Table of Contents

- [Introduction](#introduction)
- [Model Categories](#model-categories)
  - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
  - [Diffusion Models](#diffusion-models)
  - [Autoregressive Models](#autoregressive-models)
  - [Large Language Models](#large-language-models)
  - [Other Architectures](#other-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This repository documents state-of-the-art generative models that learn to create various types of data. Each model is carefully selected based on its innovation, impact, and practical applications. The documentation includes model descriptions, architectures, use cases, and relevant resources.

## Model Categories

### Generative Adversarial Networks (GANs)

| Model | Year | Paper | Code | Description |
|-------|------|-------|------|-------------|
| **StyleGAN** | 2018 | [Paper](https://arxiv.org/abs/1812.04948) | [Code](https://github.com/NVlabs/stylegan) | High-quality image generation with style-based generator architecture. |
| **BigGAN** | 2018 | [Paper](https://arxiv.org/abs/1809.11096) | [Code](https://github.com/ajbrock/BigGAN-PyTorch) | Large scale GAN training for high-fidelity natural image synthesis. |
| **CycleGAN** | 2017 | [Paper](https://arxiv.org/abs/1703.10593) | [Code](https://github.com/junyanz/CycleGAN) | Unpaired image-to-image translation using cycle-consistent adversarial networks. |

### Variational Autoencoders (VAEs)

| Model | Year | Paper | Code | Description |
|-------|------|-------|------|-------------|
| **VQ-VAE** | 2017 | [Paper](https://arxiv.org/abs/1711.00937) | [Code](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb) | Vector Quantized Variational Autoencoder with discrete latent representations. |
| **NVAE** | 2020 | [Paper](https://arxiv.org/abs/2007.03898) | [Code](https://github.com/NVlabs/NVAE) | A deep hierarchical VAE that achieves state-of-the-art results. |

### Diffusion Models

| Model | Year | Paper | Code | Description |
|-------|------|-------|------|-------------|
| **DDPM** | 2020 | [Paper](https://arxiv.org/abs/2006.11239) | [Code](https://github.com/hojonathanho/diffusion) | Denoising Diffusion Probabilistic Models for high-quality image generation. |
| **Stable Diffusion** | 2022 | [Paper](https://arxiv.org/abs/2112.10752) | [Code](https://github.com/CompVis/stable-diffusion) | Latent text-to-image diffusion model capable of generating photo-realistic images. |
| **Imagen** | 2022 | [Paper](https://arxiv.org/abs/2205.11487) | [Code](https://github.com/lucidrains/imagen-pytorch) | Photorealistic text-to-image diffusion models with deep language understanding. |

### Autoregressive Models

| Model | Year | Paper | Code | Description |
|-------|------|-------|------|-------------|
| **PixelCNN** | 2016 | [Paper](https://arxiv.org/abs/1601.06759) | [Code](https://github.com/openai/pixel-cnn) | Deep autoregressive models for image generation. |
| **WaveNet** | 2016 | [Paper](https://arxiv.org/abs/1609.03499) | [Code](https://github.com/ibab/tensorflow-wavenet) | Generative model for raw audio waveforms. |

### Large Language Models

| Model | Year | Paper | Code | Description |
|-------|------|-------|------|-------------|
| **GPT-3** | 2020 | [Paper](https://arxiv.org/abs/2005.14165) | [API](https://openai.com/api/) | Large language model with 175B parameters for text generation. |
| **GPT-4** | 2023 | [Blog](https://openai.com/research/gpt-4) | [API](https://openai.com/gpt-4) | Next-generation multimodal model with improved capabilities. |
| **LLaMA** | 2023 | [Paper](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) | [Code](https://github.com/facebookresearch/llama) | Collection of foundation language models by Meta AI. |

### Other Architectures

| Model | Year | Paper | Code | Description |
|-------|------|-------|------|-------------|
| **DALL·E** | 2021 | [Blog](https://openai.com/research/dall-e) | [API](https://openai.com/dall-e-2) | Neural network that creates images from text descriptions. |
| **CLIP** | 2021 | [Paper](https://arxiv.org/abs/2103.00020) | [Code](https://github.com/openai/CLIP) | Learns visual concepts from natural language supervision. |

## Evaluation Metrics

Different generative models are evaluated using various metrics depending on their modality:

- **Images**: 
  - FID (Fréchet Inception Distance)
  - IS (Inception Score)
  - Precision & Recall
  
- **Text**:
  - Perplexity
  - BLEU, ROUGE, METEOR
  - Human Evaluation
  
- **Audio**:
  - FAD (Fréchet Audio Distance)
  - KLD (KL Divergence)
  - MOS (Mean Opinion Score)

## Getting Started

To get started with any of these models:

1. Check the official repository for installation instructions
2. Review the model's paper for theoretical understanding
3. Look for pre-trained models when available
4. Start with provided examples or tutorials

## Contributing

Contributions to improve this repository are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your changes
3. Submit a pull request with a clear description of your updates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- All the researchers and developers who contributed to the models listed here
- The open-source community for maintaining and improving these implementations
- Special thanks to all the organizations that released their models and code publicly