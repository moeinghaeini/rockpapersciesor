"""
Data loading and preprocessing utilities for Rock-Paper-Scissors dataset.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RockPaperScissorsDataLoader:
    """
    Data loader class for Rock-Paper-Scissors dataset.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_config = self.config['data']
        self.classes = self.config['classes']
        self.raw_path = Path(self.data_config['raw_path'])
        self.processed_path = Path(self.data_config['processed_path'])
        
        # Create processed directory if it doesn't exist
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def load_dataset_info(self):
        """
        Load basic information about the dataset.
        
        Returns:
            dict: Dataset information including class counts and image paths
        """
        dataset_info = {}
        total_images = 0
        
        for class_name in self.classes:
            class_path = self.raw_path / class_name
            if class_path.exists():
                images = list(class_path.glob('*.png'))
                num_images = len(images)
                dataset_info[class_name] = {
                    'count': num_images,
                    'path': class_path,
                    'images': images
                }
                total_images += num_images
            else:
                logger.warning(f"Class directory not found: {class_path}")
                dataset_info[class_name] = {
                    'count': 0,
                    'path': class_path,
                    'images': []
                }
        
        dataset_info['total'] = total_images
        logger.info(f"Dataset loaded: {total_images} total images")
        
        return dataset_info
    
    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess a single image.
        
        Args:
            image_path (str): Path to the image
            target_size (tuple): Target size (height, width)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if target_size is None:
            target_size = tuple(self.data_config['image_size'])
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_data_generators(self, train_dir, val_dir, test_dir=None):
        """
        Create data generators for training, validation, and testing.
        
        Args:
            train_dir (str): Training data directory
            val_dir (str): Validation data directory
            test_dir (str): Test data directory (optional)
            
        Returns:
            tuple: Data generators for train, val, and test
        """
        batch_size = self.data_config['batch_size']
        target_size = tuple(self.data_config['image_size'])
        
        # Data augmentation configuration
        aug_config = self.config['augmentation']
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=aug_config['rotation_range'],
            width_shift_range=aug_config['width_shift_range'],
            height_shift_range=aug_config['height_shift_range'],
            horizontal_flip=aug_config['horizontal_flip'],
            zoom_range=aug_config['zoom_range'],
            fill_mode=aug_config['fill_mode'],
            rescale=1./255
        )
        
        # Validation and test data generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = None
        if test_dir and Path(test_dir).exists():
            test_generator = val_test_datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        return train_generator, val_generator, test_generator
    
    def split_dataset(self, dataset_info, test_size=None, val_size=None):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset_info (dict): Dataset information from load_dataset_info()
            test_size (float): Test set size (default from config)
            val_size (float): Validation set size (default from config)
            
        Returns:
            dict: Split information with paths for each set
        """
        if test_size is None:
            test_size = self.data_config['test_split']
        if val_size is None:
            val_size = self.data_config['validation_split']
        
        # Create split directories
        split_dirs = {
            'train': self.processed_path / 'train',
            'val': self.processed_path / 'val',
            'test': self.processed_path / 'test'
        }
        
        for split_dir in split_dirs.values():
            split_dir.mkdir(exist_ok=True)
            for class_name in self.classes:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        split_info = {}
        
        for class_name in self.classes:
            if class_name not in dataset_info or dataset_info[class_name]['count'] == 0:
                continue
                
            images = dataset_info[class_name]['images']
            
            # First split: separate test set
            train_val_images, test_images = train_test_split(
                images, test_size=test_size, random_state=42
            )
            
            # Second split: separate train and validation
            train_images, val_images = train_test_split(
                train_val_images, test_size=val_size/(1-test_size), random_state=42
            )
            
            split_info[class_name] = {
                'train': train_images,
                'val': val_images,
                'test': test_images
            }
            
            # Copy files to respective directories
            import shutil
            for split_name, image_list in split_info[class_name].items():
                for image_path in image_list:
                    dest_path = split_dirs[split_name] / class_name / image_path.name
                    shutil.copy2(image_path, dest_path)
            
            logger.info(f"{class_name}: train={len(train_images)}, "
                       f"val={len(val_images)}, test={len(test_images)}")
        
        return split_info, split_dirs
    
    def save_split_info(self, split_info, output_path="data/processed/split_info.yaml"):
        """
        Save split information to YAML file.
        
        Args:
            split_info (dict): Split information
            output_path (str): Output file path
        """
        # Convert Path objects to strings for YAML serialization
        serializable_info = {}
        for class_name, splits in split_info.items():
            serializable_info[class_name] = {}
            for split_name, image_paths in splits.items():
                serializable_info[class_name][split_name] = [str(p) for p in image_paths]
        
        with open(output_path, 'w') as file:
            yaml.dump(serializable_info, file, default_flow_style=False)
        
        logger.info(f"Split information saved to {output_path}")


def main():
    """
    Example usage of the data loader.
    """
    # Initialize data loader
    loader = RockPaperScissorsDataLoader()
    
    # Load dataset info
    dataset_info = loader.load_dataset_info()
    
    # Split dataset
    split_info, split_dirs = loader.split_dataset(dataset_info)
    
    # Save split information
    loader.save_split_info(split_info)
    
    # Create data generators
    train_gen, val_gen, test_gen = loader.create_data_generators(
        str(split_dirs['train']),
        str(split_dirs['val']),
        str(split_dirs['test'])
    )
    
    print("Data loading and preprocessing completed successfully!")


if __name__ == "__main__":
    main()
