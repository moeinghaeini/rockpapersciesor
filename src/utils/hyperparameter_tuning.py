"""
Hyperparameter tuning utilities for Rock-Paper-Scissors CNN project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import optuna
import yaml
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning class for CNN models.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize hyperparameter tuner with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.tuning_config = self.config['hyperparameter_tuning']
        self.results_config = self.config['results']
        self.classes = self.config['classes']
        
        # Create results directories
        os.makedirs(self.results_config['logs_path'], exist_ok=True)
    
    def grid_search(self, model_creator, train_generator, val_generator, model_name):
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model_creator: Model creation function
            train_generator: Training data generator
            val_generator: Validation data generator
            model_name (str): Name of the model
            
        Returns:
            dict: Best parameters and results
        """
        logger.info(f"Starting grid search for {model_name}")
        
        param_grid = self.tuning_config['param_grid']
        best_score = 0
        best_params = None
        results = []
        
        # Create parameter grid
        grid = ParameterGrid(param_grid)
        total_combinations = len(grid)
        
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        for i, params in enumerate(grid):
            logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            try:
                # Create model with current parameters
                model = model_creator(params)
                
                # Train model
                history = model.fit(
                    train_generator,
                    epochs=10,  # Reduced epochs for tuning
                    validation_data=val_generator,
                    verbose=0
                )
                
                # Get best validation accuracy
                best_val_acc = max(history.history['val_accuracy'])
                
                # Store results
                result = {
                    'params': params,
                    'val_accuracy': best_val_acc,
                    'val_loss': min(history.history['val_loss']),
                    'train_accuracy': max(history.history['accuracy']),
                    'train_loss': min(history.history['loss'])
                }
                results.append(result)
                
                # Update best parameters
                if best_val_acc > best_score:
                    best_score = best_val_acc
                    best_params = params
                
                logger.info(f"Validation accuracy: {best_val_acc:.4f}")
                
            except Exception as e:
                logger.error(f"Error with parameters {params}: {str(e)}")
                continue
        
        # Save results
        self._save_tuning_results(results, model_name, "grid_search")
        
        logger.info(f"Grid search completed. Best accuracy: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def random_search(self, model_creator, train_generator, val_generator, model_name):
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            model_creator: Model creation function
            train_generator: Training data generator
            val_generator: Validation data generator
            model_name (str): Name of the model
            
        Returns:
            dict: Best parameters and results
        """
        logger.info(f"Starting random search for {model_name}")
        
        param_grid = self.tuning_config['param_grid']
        n_trials = self.tuning_config['n_trials']
        best_score = 0
        best_params = None
        results = []
        
        logger.info(f"Testing {n_trials} random parameter combinations")
        
        for i in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_grid.items():
                params[param_name] = np.random.choice(param_values)
            
            logger.info(f"Testing combination {i+1}/{n_trials}: {params}")
            
            try:
                # Create model with current parameters
                model = model_creator(params)
                
                # Train model
                history = model.fit(
                    train_generator,
                    epochs=10,  # Reduced epochs for tuning
                    validation_data=val_generator,
                    verbose=0
                )
                
                # Get best validation accuracy
                best_val_acc = max(history.history['val_accuracy'])
                
                # Store results
                result = {
                    'params': params,
                    'val_accuracy': best_val_acc,
                    'val_loss': min(history.history['val_loss']),
                    'train_accuracy': max(history.history['accuracy']),
                    'train_loss': min(history.history['loss'])
                }
                results.append(result)
                
                # Update best parameters
                if best_val_acc > best_score:
                    best_score = best_val_acc
                    best_params = params
                
                logger.info(f"Validation accuracy: {best_val_acc:.4f}")
                
            except Exception as e:
                logger.error(f"Error with parameters {params}: {str(e)}")
                continue
        
        # Save results
        self._save_tuning_results(results, model_name, "random_search")
        
        logger.info(f"Random search completed. Best accuracy: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def optuna_optimization(self, model_creator, train_generator, val_generator, model_name):
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            model_creator: Model creation function
            train_generator: Training data generator
            val_generator: Validation data generator
            model_name (str): Name of the model
            
        Returns:
            dict: Best parameters and results
        """
        logger.info(f"Starting Optuna optimization for {model_name}")
        
        n_trials = self.tuning_config['n_trials']
        
        def objective(trial):
            # Define parameter ranges
            params = {}
            param_grid = self.tuning_config['param_grid']
            
            for param_name, param_values in param_grid.items():
                if param_name == 'learning_rate':
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values), log=True
                    )
                elif param_name == 'batch_size':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
                elif param_name == 'dropout':
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
            
            try:
                # Create model with current parameters
                model = model_creator(params)
                
                # Train model
                history = model.fit(
                    train_generator,
                    epochs=10,  # Reduced epochs for tuning
                    validation_data=val_generator,
                    verbose=0
                )
                
                # Return best validation accuracy
                return max(history.history['val_accuracy'])
                
            except Exception as e:
                logger.error(f"Error in trial: {str(e)}")
                return 0.0
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        
        # Save study results
        self._save_optuna_results(study, model_name)
        
        logger.info(f"Optuna optimization completed. Best accuracy: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def _save_tuning_results(self, results, model_name, method):
        """
        Save tuning results to file.
        
        Args:
            results (list): List of tuning results
            model_name (str): Name of the model
            method (str): Tuning method used
        """
        # Convert to DataFrame
        df_results = []
        for result in results:
            row = result['params'].copy()
            row.update({
                'val_accuracy': result['val_accuracy'],
                'val_loss': result['val_loss'],
                'train_accuracy': result['train_accuracy'],
                'train_loss': result['train_loss']
            })
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Save to CSV
        results_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_{method}_results.csv"
        )
        df.to_csv(results_path, index=False)
        
        # Save summary
        summary_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_{method}_summary.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write(f"Hyperparameter Tuning Results - {method.upper()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Total trials: {len(results)}\n\n")
            
            if results:
                best_result = max(results, key=lambda x: x['val_accuracy'])
                f.write(f"Best validation accuracy: {best_result['val_accuracy']:.4f}\n")
                f.write(f"Best parameters: {best_result['params']}\n\n")
                
                f.write("Top 5 results:\n")
                f.write("-" * 20 + "\n")
                sorted_results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
                for i, result in enumerate(sorted_results[:5]):
                    f.write(f"{i+1}. Accuracy: {result['val_accuracy']:.4f}, "
                           f"Params: {result['params']}\n")
        
        logger.info(f"Tuning results saved to {results_path}")
        logger.info(f"Tuning summary saved to {summary_path}")
    
    def _save_optuna_results(self, study, model_name):
        """
        Save Optuna study results.
        
        Args:
            study (optuna.Study): Optuna study object
            model_name (str): Name of the model
        """
        # Save study as pickle
        study_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_optuna_study.pkl"
        )
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        # Save summary
        summary_path = os.path.join(
            self.results_config['logs_path'],
            f"{model_name}_optuna_summary.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write(f"Optuna Optimization Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Best value: {study.best_value:.4f}\n")
            f.write(f"Best parameters: {study.best_params}\n")
            f.write(f"Number of trials: {len(study.trials)}\n\n")
            
            f.write("Top 5 trials:\n")
            f.write("-" * 15 + "\n")
            sorted_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)
            for i, trial in enumerate(sorted_trials[:5]):
                f.write(f"{i+1}. Value: {trial.value:.4f}, "
                       f"Params: {trial.params}\n")
        
        logger.info(f"Optuna study saved to {study_path}")
        logger.info(f"Optuna summary saved to {summary_path}")


def main():
    """
    Example usage of hyperparameter tuning utilities.
    """
    print("Hyperparameter tuning utilities module loaded successfully!")
    print("Use HyperparameterTuner class to perform hyperparameter optimization.")


if __name__ == "__main__":
    main()
