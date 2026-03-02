# TensorTonic 🧠

A comprehensive deep learning implementation repository focused on building neural network architectures and components from scratch using NumPy and PyTorch. This project serves as both an educational resource and a research playground for understanding the fundamental mechanics of modern deep learning architectures.

## 📋 Overview

TensorTonic is a collection of hand-crafted implementations of popular deep learning architectures and components. The goal is to provide clear, well-documented code that demonstrates how these complex systems work under the hood, without relying heavily on high-level frameworks.

**Key Features:**
- 🔬 Research-grade implementations of GANs and Transformers
- 📚 Educational code with detailed function documentation
- 🛠️ Built primarily with NumPy for transparency
- 🎯 Focus on understanding fundamentals over performance optimization

## 📂 Repository Structure

```
TensorTonic/
├── code/                    # Basic building blocks and utility functions
│   └── sigmoid.py          # Vectorized sigmoid activation function
├── research/               # Advanced neural network architectures
│   ├── GAN/               # Generative Adversarial Network implementations
│   │   ├── discriminator.py    # Discriminator network (real/fake classifier)
│   │   ├── generator.py        # Generator network (creates fake data)
│   │   └── neural_networks.py  # Core neural network building blocks
│   └── transformer/       # Transformer architecture components
│       ├── attention.py           # Scaled dot-product attention (PyTorch)
│       ├── feed_forward_network.py # Position-wise FFN
│       ├── layer_norm.py          # Layer normalization
│       ├── multihead_attention.py # Multi-head attention mechanism
│       ├── positional_encoding.py # Sinusoidal positional encodings
│       ├── tokenizer.py           # Simple word-level tokenizer
│       ├── tokens_embedding.py    # Token embedding layer (PyTorch)
│       └── transformer.py         # Complete encoder block implementation
└── README.md              # This file
```

## 🎓 Topics Covered

### 1. **Basic Neural Network Components** (`code/`)

#### Activation Functions
- **Sigmoid**: Vectorized implementation for binary classification and probability outputs

### 2. **Generative Adversarial Networks (GANs)** (`research/GAN/`)

A complete GAN implementation built from scratch with NumPy:

#### Discriminator Network
- **Architecture**: 3-layer feedforward network
- **Input Layer**: Variable dimension (depends on data)
- **Hidden Layers**: 256 → 128 neurons with LeakyReLU activation
- **Output Layer**: Single neuron with Sigmoid activation (real/fake classification)
- **Purpose**: Distinguishes between real and generated (fake) data

#### Generator Network
- **Architecture**: 3-layer feedforward network  
- **Input Layer**: Random noise vector (latent space)
- **Hidden Layers**: 128 → 256 neurons with LeakyReLU activation
- **Output Layer**: Tanh activation (generates data in [-1, 1] range)
- **Purpose**: Transforms random noise into realistic-looking data

#### Core Building Blocks (`neural_networks.py`)
- **Linear Layer**: Fully connected layer implementation (W·x + b)
- **Activation Functions**:
  - LeakyReLU: Prevents dying ReLU problem with negative slope (α=0.2)
  - Tanh: Hyperbolic tangent for output normalization
  - Sigmoid: Binary classification activation
- **Weight Initialization**: He initialization with small epsilon for numerical stability

### 3. **Transformer Architecture** (`research/transformer/`)

A comprehensive implementation of the Transformer architecture, primarily using NumPy:

#### Attention Mechanisms
- **Scaled Dot-Product Attention**: 
  - Core attention computation with scaling by √d_k
  - Available in both PyTorch (`attention.py`) and NumPy (`transformer.py`)
  - Formula: Attention(Q,K,V) = softmax(QK^T / √d_k)V

- **Multi-Head Attention**:
  - Parallel attention heads for learning diverse representations
  - Projects Q, K, V into multiple subspaces (num_heads)
  - Concatenates head outputs and applies final linear projection
  - Enables model to attend to information from different representation subspaces

#### Core Components
- **Positional Encoding**: 
  - Sinusoidal encodings to inject sequence position information
  - Uses sin for even dimensions, cos for odd dimensions
  - No learnable parameters - purely deterministic

- **Feed-Forward Network**:
  - Position-wise MLP applied to each position independently
  - Two linear transformations with ReLU activation
  - Structure: Linear → ReLU → Linear

- **Layer Normalization**:
  - Normalizes across feature dimension
  - Learnable scale (gamma) and shift (beta) parameters
  - Improves training stability and convergence

- **Token Embedding**:
  - Converts token indices to dense vectors
  - Scaled by √d_model for better gradient flow
  - PyTorch-based implementation

- **Tokenizer**:
  - Word-level tokenization with special tokens
  - Supports: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
  - Vocabulary building from text corpus
  - Bidirectional encode/decode functionality

#### Complete Encoder Block
The `transformer.py` file implements a full encoder block combining all components:
1. **Multi-Head Self-Attention** with residual connection
2. **Add & Norm** (Layer Normalization)
3. **Feed-Forward Network** with residual connection
4. **Add & Norm** (Layer Normalization)

This follows the original "Attention is All You Need" paper architecture.

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
numpy>=1.21.0
torch>=1.10.0  # For some transformer components
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ChuckySRB/TensorTonic.git
cd TensorTonic

# Install dependencies (recommended: use a virtual environment)
pip install numpy torch
```

### Usage Examples

#### Using the Sigmoid Function
```python
from code.sigmoid import sigmoid
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
output = sigmoid(x)
print(output)  # [0.119, 0.269, 0.5, 0.731, 0.881]
```

#### Using the Tokenizer
```python
from research.transformer.tokenizer import SimpleTokenizer

texts = ["Hello world", "This is a test"]
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)

encoded = tokenizer.encode("Hello world")
decoded = tokenizer.decode(encoded)
```

#### Building a GAN Component
```python
import numpy as np
from research.GAN.generator import generator
from research.GAN.discriminator import discriminator

# Generate fake data from random noise
noise = np.random.randn(32, 100)  # Batch of 32, noise dim 100
fake_data = generator(noise, output_dim=784)  # Generate 28x28 images

# Classify data as real or fake
predictions = discriminator(fake_data)
```

## 🔮 Topics To Be Covered

Future implementations and enhancements planned for TensorTonic:

### Short-term Goals
- [ ] **Complete GAN Training Loop**: End-to-end training with loss computation and optimization
- [ ] **Decoder Block**: Complete the transformer with decoder implementation
- [ ] **Full Transformer Model**: Combine encoder and decoder for seq2seq tasks
- [ ] **Attention Visualization**: Tools to visualize attention weights
- [ ] **More Activation Functions**: ReLU, ELU, Swish, GELU implementations

### Medium-term Goals
- [ ] **Convolutional Neural Networks (CNNs)**:
  - Convolution layers
  - Pooling operations
  - Classic architectures (LeNet, AlexNet, VGG, ResNet)
- [ ] **Recurrent Neural Networks (RNNs)**:
  - Vanilla RNN
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
- [ ] **Advanced GAN Variants**:
  - DCGAN (Deep Convolutional GAN)
  - WGAN (Wasserstein GAN)
  - StyleGAN components
- [ ] **Optimization Algorithms**:
  - SGD, Adam, RMSprop implementations
  - Learning rate schedulers

### Long-term Goals
- [ ] **Vision Transformer (ViT)**: Transformers for image classification
- [ ] **BERT-style Models**: Masked language modeling and pre-training
- [ ] **GPT-style Autoregressive Models**: Decoder-only transformers
- [ ] **Diffusion Models**: Score-based generative models
- [ ] **Reinforcement Learning**: Policy gradients, Q-learning, PPO
- [ ] **Graph Neural Networks**: Message passing and graph convolutions
- [ ] **Comprehensive Testing Suite**: Unit tests for all components
- [ ] **Jupyter Notebook Tutorials**: Interactive learning materials
- [ ] **Performance Benchmarks**: Compare with framework implementations

## 🤝 Contributing

Contributions are welcome! Whether you want to:
- Fix bugs or improve existing implementations
- Add new architectures or components
- Improve documentation
- Add tests or examples

Please feel free to open an issue or submit a pull request.

### Contribution Guidelines
1. Keep implementations clear and educational
2. Add comprehensive docstrings to functions
3. Follow the existing code structure
4. Test your code before submitting
5. Update documentation as needed

## 📚 Learning Resources

To better understand the implementations in this repository:

- **GANs**: [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- **Transformers**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **Deep Learning Book**: [deeplearningbook.org](http://www.deeplearningbook.org/)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

This repository is built for learning and research purposes, implementing concepts from seminal papers in deep learning and neural networks.

---

**Note**: This is an educational project focused on understanding. For production use cases, consider using optimized frameworks like PyTorch, TensorFlow, or JAX.