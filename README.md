# FungiX: Attention-Guided EfficientNetV2-S Framework for Fungal Classification

<div align="center">

![FungiX](https://img.shields.io/badge/Deep%20Learning-Fungal%20Classification-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

**A robust and explainable deep learning framework for microscopic fungal image classification using attention-guided EfficientNetV2-S architecture.**

</div>

---

## 📋 Overview

FungiX is an advanced machine learning framework designed for automated classification of microscopic fungal images. By combining the efficiency of EfficientNetV2-S with attention mechanisms, FungiX delivers:

- **High Accuracy**: State-of-the-art classification performance on fungal microscopy datasets
- **Robustness**: Attention-guided mechanisms for focusing on discriminative features
- **Explainability**: Visual attention maps for model interpretability
- **Efficiency**: Lightweight architecture suitable for deployment

This framework is ideal for mycology research, clinical diagnostics, and automated fungal identification in laboratory settings.

---

## 🎯 Features

- ✅ **Attention-Guided Architecture**: Incorporates channel and spatial attention mechanisms to highlight relevant fungal features
- ✅ **EfficientNetV2-S Backbone**: Optimized for accuracy-efficiency trade-off
- ✅ **Multi-class Classification**: Support for diverse fungal species identification
- ✅ **Explainable Predictions**: Attention weight visualization for model interpretability
- ✅ **Data Augmentation**: Comprehensive preprocessing and augmentation strategies
- ✅ **Easy Integration**: Jupyter notebook-based implementation for seamless workflow
- ✅ **Transfer Learning**: Leverage pre-trained weights for improved performance
- ✅ **Reproducible Research**: Clear documentation and publicly available datasets

---

## 👥 Authors

**Research Team:**

1. **Jonayed Al-Faruk** - Department of Computer Science and Engineering, Mymensingh Engineering College, Mymensingh-2200, Bangladesh
   - Email: [jonayedalfaruk211282@gmail.com](mailto:jonayedalfaruk211282@gmail.com)

2. **Md Rifat Hossen** (Corresponding Author) - Department of Information and Communication Engineering, Pabna University of Science and Technology, Pabna-6600, Bangladesh
   - Email: [rifat.220614@s.pust.ac.bd](mailto:rifat.220614@s.pust.ac.bd)

3. **Ishtiaq Ahammad** - Department of Information and Communication Engineering, Noakhali Science and Technology University, Noakhali-3814, Chattogram, Bangladesh
   - Email: [ishtiaq@nstu.edu.bd](mailto:ishtiaq@nstu.edu.bd)

4. **Sakiba Sarkar** - Department of Computer Science and Engineering, Mymensingh Engineering College, Mymensingh-2200, Bangladesh
   - Email: [sakibasarkar1999@gmail.com](mailto:sakibasarkar1999@gmail.com)

5. **Joyonta Das** - Department of Electrical and Electronic Engineering, Mymensingh Engineering College, Mymensingh-2200, Bangladesh
   - Email: [joyontadas.mec@gmail.com](mailto:joyontadas.mec@gmail.com)

---

## 🏗️ Architecture

### Model Components

```
Input (Microscopic Image)
        ↓
EfficientNetV2-S Backbone (Feature Extraction)
        ↓
Attention Module (Channel & Spatial)
        ↓
Feature Refinement
        ↓
Classification Head
        ↓
Output (Fungal Species Prediction)
```

### Key Design Principles

1. **Feature Extraction**: EfficientNetV2-S efficiently extracts hierarchical features from fungal images
2. **Attention Mechanism**: Learns to focus on discriminative fungal morphological characteristics
3. **Robust Classification**: Multi-headed classification for reliable predictions
4. **Interpretability**: Attention maps provide visual explanations for model decisions

---

## 📂 Repository Structure

```
FungiX/
├── defungi1.ipynb              # Main training and evaluation notebook
├── defungi23.ipynb             # Additional experiments and analysis
├── README.md                   # This file
└── [Data and model artifacts]
```

### Notebook Contents

- **defungi1.ipynb**: 
  - Data loading and preprocessing
  - Model architecture definition
  - Training pipeline
  - Validation and testing procedures
  - Attention visualization

- **defungi23.ipynb**: 
  - Extended experiments
  - Cross-validation analysis
  - Additional visualization and analysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook/JupyterLab
- GPU (NVIDIA CUDA-compatible) recommended for training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MdRifat-Hossen/FungiX-An-Attention-Guided-EfficientNetV2-S-Framework-for-Robust-and-Explainable-Microscopic-Fungal.git
   cd FungiX-An-Attention-Guided-EfficientNetV2-S-Framework-for-Robust-and-Explainable-Microscopic-Fungal
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision pytorch-cuda=12.1
   pip install timm  # EfficientNet implementations
   pip install numpy pandas scikit-learn matplotlib seaborn
   pip install jupyter notebook
   ```

### Running the Model

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Launch the main notebook**
   - Open `defungi1.ipynb` in your browser

3. **Follow the notebook cells**
   - Data loading and preprocessing
   - Model training
   - Evaluation and visualization
   - Attention map generation

---

## 📊 Datasets

This research utilizes publicly available datasets for reproducibility and validation. The datasets used in this study are:

### Dataset 1: Microscopic Fungi Images
- **Source**: [Kaggle - Microscopic Fungi Images](https://www.kaggle.com/datasets/anshtanwar/microscopic-fungi-images)
- **Description**: Large-scale collection of microscopic fungal images with labeled species
- **Access**: Freely available on Kaggle platform

### Dataset 2: DeFungi Dataset
- **Source**: [UCI Machine Learning Repository - DeFungi](https://archive.ics.uci.edu/dataset/773/defungi)
- **Description**: Comprehensive defungification dataset with diverse fungal species
- **Access**: Publicly available from UCI ML Archive

### Downloading Datasets

1. **For Kaggle Dataset**:
   ```bash
   # Install kaggle CLI
   pip install kaggle
   
   # Download dataset (requires Kaggle API key)
   kaggle datasets download -d anshtanwar/microscopic-fungi-images
   ```

2. **For UCI Dataset**:
   ```bash
   # Visit https://archive.ics.uci.edu/dataset/773/defungi
   # Download directly or use automated script
   ```

---

## 📝 Usage Guide

### Basic Training

```python
# Initialize model
model = FungiXAttentionNet(num_classes=num_fungal_species)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference and Attention Visualization

```python
# Load trained model
model.load_state_dict(torch.load('fungix_model.pth'))
model.eval()

# Make prediction with attention
image = preprocess(fungal_image)
output, attention_map = model(image, return_attention=True)

# Visualize attention
visualize_attention(fungal_image, attention_map)
```

---

## 📈 Performance

Expected performance metrics on standard fungal microscopy datasets:

| Metric | Value |
|--------|-------|
| Accuracy | >95% |
| Precision | >94% |
| Recall | >93% |
| F1-Score | >93% |

*Note: Actual performance depends on dataset size and quality*

---

## 🔬 Supported Fungal Species

The framework can be trained on various fungal datasets including:

- Common molds and yeasts
- Pathogenic fungi
- Environmental fungi
- Clinical isolates
- Research species

---

## 📚 Dataset Preparation

### Expected Format

```
data/
├── train/
│   ├── Aspergillus/
│   ├── Candida/
│   └── Penicillium/
├── val/
│   ├── Aspergillus/
│   ├── Candida/
│   └── Penicillium/
└── test/
    ├── Aspergillus/
    ├── Candida/
    └── Penicillium/
```

### Data Augmentation

- Random rotation (±15°)
- Horizontal/vertical flipping
- Color jittering
- Gaussian blur
- Elastic deformations

---

## 🎨 Attention Visualization

FungiX generates interpretable attention maps showing which regions of microscopic images influence predictions:

```python
# Generate attention visualization
attention_maps = model.get_attention_maps(test_images)
plot_attention_heatmaps(test_images, attention_maps)
```

---

## 🛠️ Configuration

Key hyperparameters can be customized:

```python
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100,
    'dropout_rate': 0.3,
    'attention_heads': 8,
    'image_size': 224,
    'num_classes': 10,  # Number of fungal species
}
```

---

## 🔄 Reproducibility

### Code Availability

The complete code implementation is available in this repository. For additional materials, code variations, or extended documentation beyond what is published, please contact the corresponding author:

- **Corresponding Author**: Md Rifat Hossen
- **Email**: [rifat.220614@s.pust.ac.bd](mailto:rifat.220614@s.pust.ac.bd)

### Reproducibility Guidelines

1. **Dataset Access**: All datasets used are publicly available (see [Datasets](#-datasets) section)
2. **Environment**: Use the provided `requirements.txt` for dependency management
3. **Hyperparameters**: All model hyperparameters are documented in configuration sections
4. **Random Seeds**: Set random seeds for reproducible results:
   ```python
   import torch
   import numpy as np
   torch.manual_seed(42)
   np.random.seed(42)
   ```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests for:

- Bug fixes
- Performance improvements
- New attention mechanisms
- Additional visualization tools
- Documentation enhancements

---

## 📜 Citation

If you use FungiX in your research, please cite this work:

```bibtex
@article{hossen2024fungix,
  title={FungiX: An Attention-Guided EfficientNetV2-S Framework for Robust and Explainable Microscopic Fungal Classification},
  author={Al-Faruk, Jonayed and Hossen, Md Rifat and Ahammad, Ishtiaq and Sarkar, Sakiba and Das, Joyonta},
  year={2024},
  url={https://github.com/MdRifat-Hossen/FungiX-An-Attention-Guided-EfficientNetV2-S-Framework-for-Robust-and-Explainable-Microscopic-Fungal}
}
```

### Alternative Citation Formats

**APA Format:**
```
Al-Faruk, J., Hossen, M. R., Ahammad, I., Sarkar, S., & Das, J. (2024). FungiX: 
An Attention-Guided EfficientNetV2-S Framework for Robust and Explainable Microscopic 
Fungal Classification. GitHub Repository.
```

**IEEE Format:**
```
[1] J. Al-Faruk, M. R. Hossen, I. Ahammad, S. Sarkar, and J. Das, "FungiX: An 
Attention-Guided EfficientNetV2-S Framework for Robust and Explainable Microscopic 
Fungal Classification," 2024. [Online]. Available: 
https://github.com/MdRifat-Hossen/FungiX-An-Attention-Guided-EfficientNetV2-S-Framework-for-Robust-and-Explainable-Microscopic-Fungal
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Contact & Support

- **Repository Issues**: [Report a bug or request a feature](https://github.com/MdRifat-Hossen/FungiX-An-Attention-Guided-EfficientNetV2-S-Framework-for-Robust-and-Explainable-Microscopic-Fungal/issues)

- **Corresponding Author**:
  - **Name**: Md Rifat Hossen
  - **Affiliation**: Department of Information and Communication Engineering, Pabna University of Science and Technology
  - **Email**: [rifat.220614@s.pust.ac.bd](mailto:rifat.220614@s.pust.ac.bd)

- **Lead Author**:
  - **Name**: Jonayed Al-Faruk
  - **Affiliation**: Department of Computer Science and Engineering, Mymensingh Engineering College
  - **Email**: [jonayedalfaruk211282@gmail.com](mailto:jonayedalfaruk211282@gmail.com)

---

## 🙏 Acknowledgments

- EfficientNetV2 architecture by Google Research
- PyTorch team for excellent deep learning framework
- Attention mechanism inspired by recent computer vision research
- Mycology research community for fungal datasets
- Contributors from:
  - Mymensingh Engineering College, Bangladesh
  - Pabna University of Science and Technology, Bangladesh
  - Noakhali Science and Technology University, Bangladesh

---

## 📖 Related Resources

- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.14294)
- [Attention Mechanisms in Vision](https://arxiv.org/abs/1906.04341)
- [Transfer Learning Best Practices](https://cs231n.github.io/transfer-learning/)

---

<div align="center">

**Made with ❤️ by the FungiX Research Team**

For questions, collaboration inquiries, or dataset requests, please contact the corresponding author.

[![GitHub](https://img.shields.io/badge/GitHub-View_Repository-black?logo=github)](https://github.com/MdRifat-Hossen/FungiX-An-Attention-Guided-EfficientNetV2-S-Framework-for-Robust-and-Explainable-Microscopic-Fungal)

</div>
