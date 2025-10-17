# Rock-Paper-Scissors CNN Classification

A comprehensive machine learning project for classifying hand gestures in the Rock-Paper-Scissors game using Convolutional Neural Networks (CNNs).

## 📋 Project Overview

This project implements a CNN-based classifier for the Rock-Paper-Scissors game, following sound machine learning practices including proper data preprocessing, model architecture design, hyperparameter tuning, and comprehensive evaluation.

### 🎯 Objectives

- Develop CNN architectures with increasing complexity (Simple, Medium, Complex)
- Implement proper data preprocessing and augmentation
- Perform systematic hyperparameter tuning
- Evaluate model performance using multiple metrics
- Analyze model behavior and misclassifications

## 🏗️ Project Structure

```
rockpapersciesor/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw dataset (download from Kaggle)
│   └── processed/               # Processed data and splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_evaluation_analysis.ipynb
├── src/
│   ├── data/
│   │   └── data_loader.py       # Data loading and preprocessing
│   ├── models/
│   │   └── cnn_models.py        # CNN model definitions
│   └── utils/
│       ├── training_utils.py    # Training and evaluation utilities
│       └── hyperparameter_tuning.py  # Hyperparameter optimization
├── results/
│   ├── models/                  # Saved models
│   ├── plots/                   # Visualization plots
│   └── logs/                    # Training logs and results
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rockpapersciesor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

1. Download the Rock-Paper-Scissors dataset from [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
2. Extract the dataset to `data/raw/` directory
3. The structure should be:
   ```
   data/raw/
   ├── rock/
   ├── paper/
   └── scissors/
   ```

### 3. Configuration

Edit `config/config.yaml` to adjust:
- Image size and batch size
- Model architectures
- Training parameters
- Hyperparameter tuning settings

### 4. Running the Project

Execute the notebooks in order:

```bash
# Start Jupyter Lab
jupyter lab

# Or run individual notebooks
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📊 Model Architectures

### Simple CNN
- 2 convolutional layers (32, 64 filters)
- Max pooling and dropout
- Single dense layer (128 units)

### Medium CNN
- 3 convolutional layers (32, 64, 128 filters)
- Batch normalization
- Dropout regularization
- Dense layer (256 units)

### Complex CNN
- 4 convolutional layers (32, 64, 128, 256 filters)
- Batch normalization
- Global average pooling
- Multiple dense layers (512, 256 units)
- Advanced dropout strategy

## 🔧 Features

### Data Processing
- Image resizing and normalization
- Data augmentation (rotation, shift, zoom, flip)
- Proper train/validation/test splitting
- Data quality assessment

### Model Training
- Multiple CNN architectures
- Early stopping and learning rate reduction
- Model checkpointing
- Training history visualization

### Hyperparameter Tuning
- Grid search
- Random search
- Optuna optimization
- Cross-validation support

### Evaluation
- Multiple metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Classification report
- Misclassification analysis
- Training curve plots

## 📈 Results

The project evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🛠️ Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing
- **Optuna**: Hyperparameter optimization
- **Jupyter**: Interactive development

## 📝 Usage Examples

### Data Loading
```python
from src.data.data_loader import RockPaperScissorsDataLoader

loader = RockPaperScissorsDataLoader()
dataset_info = loader.load_dataset_info()
```

### Model Creation
```python
from src.models.cnn_models import RockPaperScissorsCNN

cnn_creator = RockPaperScissorsCNN()
simple_model = cnn_creator.create_simple_cnn()
```

### Training
```python
from src.utils.training_utils import TrainingManager

trainer = TrainingManager()
history = trainer.train_model(model, train_gen, val_gen, "simple_cnn")
```

### Hyperparameter Tuning
```python
from src.utils.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner()
results = tuner.grid_search(model_creator, train_gen, val_gen, "simple_cnn")
```

## 📚 Academic Context

This project is part of the Machine Learning course (A.Y. 2024/25) taught by Prof. Nicolò Cesa-Bianchi at the University of Milan.

### Evaluation Criteria
- Correctness and completeness of methodology
- Reproducibility of results
- Quality of analysis and report
- Sound statistical practices

### Key Requirements
- At least 3 CNN architectures with increasing complexity
- Proper data preprocessing and augmentation
- Systematic hyperparameter tuning
- Comprehensive evaluation and analysis
- 10-15 page detailed report

## 🤝 Contributing

This is an academic project. For questions or issues, please contact the teaching assistants:
- Luigi Foscari (luigi.foscari@unimi.it)
- Emmanuel Esposito (emmanuel.esposito@unimi.it)

## 📄 License

This project is for educational purposes as part of the Machine Learning course curriculum.

## 🔗 References

- [Rock-Paper-Scissors Dataset on Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)