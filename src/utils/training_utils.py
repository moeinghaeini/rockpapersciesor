"""
Training utilities for Rock-Paper-Scissors CNN project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages model training, evaluation, and visualization.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize training manager with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.training_config = self.config['training']
        self.results_config = self.config['results']
        self.classes = self.config['classes']
        
        # Create results directories
        self._create_results_directories()
    
    def _create_results_directories(self):
        """Create necessary directories for results."""
        directories = [
            self.results_config['models_path'],
            self.results_config['plots_path'],
            self.results_config['logs_path']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_callbacks(self, model_name):
        """
        Get training callbacks.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Early stopping
        if self.training_config['early_stopping']['patience'] > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping']['patience'],
                restore_best_weights=self.training_config['early_stopping']['restore_best_weights'],
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        if self.training_config['reduce_lr']['patience'] > 0:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['reduce_lr']['factor'],
                patience=self.training_config['reduce_lr']['patience'],
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.results_config['models_path'], 
            f"{model_name}_best.h5"
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # CSV logger
        log_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_training_log.csv"
        )
        csv_logger = keras.callbacks.CSVLogger(log_path)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train_model(self, model, train_generator, val_generator, model_name):
        """
        Train a model.
        
        Args:
            model (keras.Model): Model to train
            train_generator: Training data generator
            val_generator: Validation data generator
            model_name (str): Name of the model
            
        Returns:
            keras.History: Training history
        """
        logger.info(f"Starting training for {model_name}")
        
        # Get callbacks
        callbacks = self.get_callbacks(model_name)
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=self.training_config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed for {model_name}")
        
        # Save training history
        if self.results_config['save_training_history']:
            self.save_training_history(history, model_name)
        
        return history
    
    def save_training_history(self, history, model_name):
        """
        Save training history to file.
        
        Args:
            history (keras.History): Training history
            model_name (str): Name of the model
        """
        history_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_history.npy"
        )
        np.save(history_path, history.history)
        logger.info(f"Training history saved to {history_path}")
    
    def plot_training_history(self, history, model_name, save_plot=True):
        """
        Plot training history.
        
        Args:
            history (keras.History): Training history
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(
                self.results_config['plots_path'],
                f"{model_name}_training_history.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {plot_path}")
        
        return fig
    
    def evaluate_model(self, model, test_generator, model_name):
        """
        Evaluate model on test set.
        
        Args:
            model (keras.Model): Model to evaluate
            test_generator: Test data generator
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {model_name} on test set")
        
        # Get predictions
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        
        # Classification report
        class_report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.classes,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_classes': true_classes,
            'predicted_classes': predicted_classes
        }
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm, model_name, save_plot=True):
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_plot:
            plot_path = os.path.join(
                self.results_config['plots_path'],
                f"{model_name}_confusion_matrix.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {plot_path}")
        
        return plt.gcf()
    
    def plot_classification_report(self, class_report, model_name, save_plot=True):
        """
        Plot classification report as heatmap.
        
        Args:
            class_report (dict): Classification report
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        # Extract metrics for each class
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        
        for class_name in self.classes:
            row = [class_report[class_name][metric] for metric in metrics]
            data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=metrics, yticklabels=self.classes)
        plt.title(f'{model_name} - Classification Report')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        
        if save_plot:
            plot_path = os.path.join(
                self.results_config['plots_path'],
                f"{model_name}_classification_report.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification report plot saved to {plot_path}")
        
        return plt.gcf()
    
    def analyze_misclassifications(self, results, model_name, test_generator, num_samples=10):
        """
        Analyze misclassified samples.
        
        Args:
            results (dict): Evaluation results
            model_name (str): Name of the model
            test_generator: Test data generator
            num_samples (int): Number of misclassified samples to show
            
        Returns:
            list: List of misclassified sample information
        """
        true_classes = results['true_classes']
        predicted_classes = results['predicted_classes']
        predictions = results['predictions']
        
        # Find misclassified samples
        misclassified_indices = np.where(true_classes != predicted_classes)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassified samples found!")
            return []
        
        # Get sample misclassifications
        sample_indices = misclassified_indices[:num_samples]
        misclassified_samples = []
        
        for idx in sample_indices:
            true_class = self.classes[true_classes[idx]]
            predicted_class = self.classes[predicted_classes[idx]]
            confidence = np.max(predictions[idx])
            
            misclassified_samples.append({
                'index': idx,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    self.classes[i]: predictions[idx][i] 
                    for i in range(len(self.classes))
                }
            })
        
        logger.info(f"Found {len(misclassified_indices)} misclassified samples")
        logger.info(f"Showing {len(sample_indices)} samples:")
        
        for sample in misclassified_samples:
            logger.info(f"  Sample {sample['index']}: {sample['true_class']} -> "
                       f"{sample['predicted_class']} (confidence: {sample['confidence']:.3f})")
        
        return misclassified_samples
    
    def save_results_summary(self, results, model_name):
        """
        Save results summary to file.
        
        Args:
            results (dict): Evaluation results
            model_name (str): Name of the model
        """
        summary_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_results_summary.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write(f"Results Summary for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
            f.write(f"Test Loss: {results['test_loss']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write("-" * 20 + "\n")
            for class_name in self.classes:
                report = results['classification_report'][class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {report['precision']:.4f}\n")
                f.write(f"  Recall: {report['recall']:.4f}\n")
                f.write(f"  F1-score: {report['f1-score']:.4f}\n\n")
        
        logger.info(f"Results summary saved to {summary_path}")


def main():
    """
    Example usage of training utilities.
    """
    # This would be used in conjunction with data loading and model creation
    print("Training utilities module loaded successfully!")
    print("Use TrainingManager class to manage model training and evaluation.")


if __name__ == "__main__":
    main()
