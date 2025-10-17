"""
CNN model definitions for Rock-Paper-Scissors classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml
import logging

logger = logging.getLogger(__name__)


class RockPaperScissorsCNN:
    """
    CNN model class for Rock-Paper-Scissors classification.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the CNN model with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_configs = self.config['models']
        self.training_config = self.config['training']
        self.classes = self.config['classes']
        self.num_classes = len(self.classes)
        
    def create_simple_cnn(self, input_shape=(224, 224, 3)):
        """
        Create a simple CNN architecture.
        
        Args:
            input_shape (tuple): Input image shape
            
        Returns:
            keras.Model: Compiled model
        """
        config = self.model_configs['simple_cnn']
        
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(config['filters'][0], config['kernel_size'], 
                         activation=config['activation'], input_shape=input_shape),
            layers.MaxPooling2D(2),
            
            # Second convolutional block
            layers.Conv2D(config['filters'][1], config['kernel_size'], 
                         activation=config['activation']),
            layers.MaxPooling2D(2),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(config['dropout']),
            layers.Dense(config['dense_units'], activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model, "Simple CNN")
    
    def create_medium_cnn(self, input_shape=(224, 224, 3)):
        """
        Create a medium complexity CNN architecture.
        
        Args:
            input_shape (tuple): Input image shape
            
        Returns:
            keras.Model: Compiled model
        """
        config = self.model_configs['medium_cnn']
        
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(config['filters'][0], config['kernel_size'], 
                         activation=config['activation'], input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Second convolutional block
            layers.Conv2D(config['filters'][1], config['kernel_size'], 
                         activation=config['activation']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Third convolutional block
            layers.Conv2D(config['filters'][2], config['kernel_size'], 
                         activation=config['activation']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(config['dropout']),
            layers.Dense(config['dense_units'], activation='relu'),
            layers.Dropout(config['dropout'] * 0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model, "Medium CNN")
    
    def create_complex_cnn(self, input_shape=(224, 224, 3)):
        """
        Create a complex CNN architecture.
        
        Args:
            input_shape (tuple): Input image shape
            
        Returns:
            keras.Model: Compiled model
        """
        config = self.model_configs['complex_cnn']
        
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(config['filters'][0], config['kernel_size'], 
                         activation=config['activation'], input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Second convolutional block
            layers.Conv2D(config['filters'][1], config['kernel_size'], 
                         activation=config['activation']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Third convolutional block
            layers.Conv2D(config['filters'][2], config['kernel_size'], 
                         activation=config['activation']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Fourth convolutional block
            layers.Conv2D(config['filters'][3], config['kernel_size'], 
                         activation=config['activation']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Global average pooling instead of flatten
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config['dropout']),
            layers.Dense(config['dense_units'], activation='relu'),
            layers.Dropout(config['dropout'] * 0.5),
            layers.Dense(config['dense_units'] // 2, activation='relu'),
            layers.Dropout(config['dropout'] * 0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model, "Complex CNN")
    
    def create_custom_cnn(self, architecture_config, input_shape=(224, 224, 3)):
        """
        Create a custom CNN based on provided configuration.
        
        Args:
            architecture_config (dict): Architecture configuration
            input_shape (tuple): Input image shape
            
        Returns:
            keras.Model: Compiled model
        """
        model = keras.Sequential()
        
        # Add input layer
        model.add(layers.Input(shape=input_shape))
        
        # Add convolutional blocks
        for i, (filters, kernel_size) in enumerate(zip(
            architecture_config.get('filters', [32, 64, 128]),
            architecture_config.get('kernel_sizes', [3, 3, 3])
        )):
            model.add(layers.Conv2D(filters, kernel_size, 
                                   activation=architecture_config.get('activation', 'relu')))
            
            if architecture_config.get('use_batch_norm', True):
                model.add(layers.BatchNormalization())
            
            if architecture_config.get('use_dropout_conv', False):
                model.add(layers.Dropout(0.1))
            
            model.add(layers.MaxPooling2D(2))
        
        # Add global pooling
        if architecture_config.get('use_global_pooling', False):
            model.add(layers.GlobalAveragePooling2D())
        else:
            model.add(layers.Flatten())
        
        # Add dense layers
        for units in architecture_config.get('dense_units', [256, 128]):
            model.add(layers.Dense(units, activation='relu'))
            if architecture_config.get('use_dropout_dense', True):
                model.add(layers.Dropout(architecture_config.get('dropout', 0.3)))
        
        # Add output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return self._compile_model(model, "Custom CNN")
    
    def _compile_model(self, model, model_name):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            model (keras.Model): Model to compile
            model_name (str): Name of the model
            
        Returns:
            keras.Model: Compiled model
        """
        # Get optimizer
        optimizer_name = self.training_config['optimizer'].lower()
        learning_rate = self.training_config['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.training_config['loss_function'],
            metrics=self.training_config['metrics']
        )
        
        logger.info(f"{model_name} compiled successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def get_model_summary(self, model):
        """
        Get model summary information.
        
        Args:
            model (keras.Model): Model to summarize
            
        Returns:
            str: Model summary
        """
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def save_model(self, model, model_name, save_path="results/models/"):
        """
        Save model to disk.
        
        Args:
            model (keras.Model): Model to save
            model_name (str): Name for the saved model
            save_path (str): Directory to save the model
        """
        import os
        
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, f"{model_name}.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")


def main():
    """
    Example usage of the CNN models.
    """
    # Initialize model creator
    cnn_creator = RockPaperScissorsCNN()
    
    # Create different models
    simple_model = cnn_creator.create_simple_cnn()
    medium_model = cnn_creator.create_medium_cnn()
    complex_model = cnn_creator.create_complex_cnn()
    
    # Print model summaries
    print("Simple CNN Summary:")
    print(cnn_creator.get_model_summary(simple_model))
    
    print("\nMedium CNN Summary:")
    print(cnn_creator.get_model_summary(medium_model))
    
    print("\nComplex CNN Summary:")
    print(cnn_creator.get_model_summary(complex_model))


if __name__ == "__main__":
    main()
