# PyTorch Deep Learning Collection

A comprehensive collection of PyTorch implementations covering fundamental concepts, neural networks, CNNs, and advanced optimization techniques. This repository serves as both a learning resource and a practical toolkit for deep learning practitioners.

## 📚 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📋 Prerequisites](#-prerequisites)
- [🛠️ Installation](#️-installation)
- [📖 Notebooks Overview](#-notebooks-overview)
- [🎯 Projects](#-projects)
- [📊 Results & Benchmarks](#-results--benchmarks)
- [🔧 Configuration](#-configuration)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd Pytorch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Jupyter Notebook/Lab
- Git

## 🛠️ Installation

### Method 1: Using requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation

```bash
pip install torch torchvision pandas scikit-learn optuna matplotlib numpy
```

### GPU Support

For CUDA support, install the appropriate PyTorch version:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Notebooks Overview

### 🧪 Fundamental Concepts

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `tensors.ipynb` | PyTorch tensor operations and basics | Tensor creation, operations, GPU acceleration |
| `autograd.ipynb` | Automatic differentiation and gradients | Computational graphs, backpropagation |
| `pytorch_nn_module.ipynb` | Neural network module system | `nn.Module`, layers, parameters |

### 🏗️ Training Pipeline Components

| Notebook | Description | Components |
|----------|-------------|------------|
| `dataset_and_dataloader_demo.ipynb` | Custom datasets and data loading | `Dataset`, `DataLoader`, batching |
| `training_pipeline.ipynb` | Basic training pipeline | Loss functions, optimizers, training loops |
| `training_pipeline_nn.ipynb` | Neural network training pipeline | Model definition, training, evaluation |
| `training_pipeline_nn_dataset_dataloader.ipynb` | Complete pipeline with datasets | End-to-end training with custom datasets |

### 🎯 Applied Projects

| Notebook | Dataset | Model | Performance |
|----------|---------|-------|-------------|
| `ann_fashion_pytorch.ipynb` | Fashion-MNIST | Feedforward ANN | ~85% accuracy |
| `ann_fashion_pytorch_gpu.ipynb` | Fashion-MNIST | ANN with GPU | ~85% accuracy |
| `ann_fashion_pytorch_optimization.ipynb` | Fashion-MNIST | Optimized ANN | ~87% accuracy |
| `ann_fashion_pytorch_gpu_optuna.ipynb` | Fashion-MNIST | ANN + Optuna tuning | **~89% accuracy** |
| `cnn_fashion_pytorch_gpu.ipynb` | Fashion-MNIST | CNN with GPU | **~93% accuracy** |

## 🎯 Projects

### 1. Fashion-MNIST Classification

**Dataset**: 70,000 grayscale images (28×28) of fashion items across 10 classes

#### Architectures
- **Feedforward ANN**: Fully connected layers with BatchNorm, ReLU, Dropout
- **CNN**: Convolutional layers with pooling, followed by dense layers

#### Optimization Techniques
- **Hyperparameter Tuning**: Optuna framework for automated optimization
- **GPU Acceleration**: CUDA-enabled training with `pin_memory=True`
- **Advanced Features**: Batch normalization, dropout regularization

#### Performance Comparison
| Model | Test Accuracy | Training Time | Parameters |
|-------|---------------|---------------|------------|
| Basic ANN | ~85% | ~5 min | ~100K |
| Optimized ANN | ~89% | ~8 min | ~200K |
| CNN | **~93%** | ~12 min | ~50K |

### 2. Breast Cancer Detection

**Dataset**: Wisconsin Diagnostic Breast Cancer Dataset

- **Features**: 30 clinical features
- **Classes**: Malignant vs Benign
- **Model**: Simple neural network for binary classification
- **Performance**: ~94% accuracy

## 📊 Results & Benchmarks

### Fashion-MNIST Optuna Optimization

**Best Hyperparameters** (from 10-trial study):
```python
{
    'num_hidden_layers': 5,
    'neurons_per_layer': 120,
    'epochs': 30,
    'learning_rate': 0.0405,
    'dropout_rate': 0.2,
    'batch_size': 64,
    'optimizer': 'SGD',
    'weight_decay': 2.40e-05
}
```

**Search Space**:
| Parameter | Range | Type |
|-----------|-------|------|
| Hidden layers | 1-5 | Integer |
| Neurons per layer | 8-128 | Integer (step=8) |
| Epochs | 10-50 | Integer (step=10) |
| Learning rate | 1e-5 - 1e-1 | Log scale |
| Dropout rate | 0.1-0.5 | Float (step=0.1) |
| Batch size | [16, 32, 64, 128] | Categorical |
| Optimizer | [Adam, SGD, RMSprop] | Categorical |
| Weight decay | 1e-5 - 1e-3 | Log scale |

### CNN Architecture Details

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 10)
        )
```

## 🔧 Configuration

### Environment Setup

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Customization Options

- **Batch Size**: Adjust based on GPU memory
- **Learning Rate**: Use learning rate schedulers for better convergence
- **Architecture**: Modify layer sizes and depth
- **Optimization**: Experiment with different optimizers and regularization

### Performance Tips

1. **GPU Usage**: Always move data and models to GPU when available
2. **Batch Size**: Larger batches often improve GPU utilization
3. **Mixed Precision**: Use `torch.cuda.amp` for faster training
4. **Data Loading**: Use `pin_memory=True` and `num_workers > 0`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex operations
- Include performance benchmarks for new models
- Update documentation for new features

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Fashion-MNIST dataset from Zalando Research
- Optuna hyperparameter optimization framework
- PyTorch team for the excellent deep learning framework

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the notebook comments for detailed explanations
- Refer to PyTorch official documentation

---

**Happy Learning! 🎓**
