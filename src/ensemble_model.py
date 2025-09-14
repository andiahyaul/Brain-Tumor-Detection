"""
Brain Tumor Detection Ensemble Model
Achieves 81.58% accuracy through weighted ensemble of two trained models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pickle
import os


class BrainTumorEnsemble:
    """
    Ensemble model combining original and fine-tuned brain tumor detection models.
    Achieves 81.58% accuracy with optimized threshold and weighted averaging.
    """

    def __init__(self, model_paths=None, weights=None, threshold=0.55):
        """
        Initialize ensemble model.

        Args:
            model_paths (list): Paths to individual models
            weights (list): Weights for each model in ensemble
            threshold (float): Classification threshold for predictions
        """
        self.model_paths = model_paths or [
            'models/brain_tumor_detector_best.h5',
            'models/brain_tumor_finetuned_best.h5'
        ]
        self.weights = weights or [0.492, 0.508]  # Optimized weights from trainv2.ipynb
        self.threshold = threshold
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load individual models and compile them."""
        for i, path in enumerate(self.model_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}")

            model = load_model(path, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            self.models[f'model_{i}'] = model

        print(f"Loaded {len(self.models)} models for ensemble")

    def predict(self, X, return_probabilities=False):
        """
        Make ensemble predictions.

        Args:
            X (numpy.ndarray): Input images
            return_probabilities (bool): Return probabilities instead of binary predictions

        Returns:
            numpy.ndarray: Predictions (binary or probabilities)
        """
        if len(X.shape) == 3:  # Convert grayscale to RGB if needed
            X = np.stack([X]*3, axis=-1)

        weighted_predictions = []

        for i, (model_key, model) in enumerate(self.models.items()):
            pred = model.predict(X, verbose=0)
            weighted_pred = pred * self.weights[i]
            weighted_predictions.append(weighted_pred)

        # Combine weighted predictions
        ensemble_probabilities = np.sum(weighted_predictions, axis=0)

        if return_probabilities:
            return ensemble_probabilities
        else:
            return (ensemble_probabilities > self.threshold).astype(int)

    def evaluate(self, X, y_true, verbose=True):
        """
        Evaluate ensemble performance.

        Args:
            X (numpy.ndarray): Input images
            y_true (numpy.ndarray): True labels
            verbose (bool): Print results

        Returns:
            dict: Performance metrics
        """
        probabilities = self.predict(X, return_probabilities=True)
        predictions = (probabilities > self.threshold).astype(int).flatten()
        y_true = y_true.flatten()

        accuracy = accuracy_score(y_true, predictions)
        try:
            auc = roc_auc_score(y_true, probabilities.flatten())
        except:
            auc = 0.0

        results = {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': predictions,
            'probabilities': probabilities,
            'threshold': self.threshold
        }

        if verbose:
            print(f"Ensemble Performance:")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  AUC: {auc:.4f}")
            print(f"  Threshold: {self.threshold}")
            print("\nClassification Report:")
            print(classification_report(y_true, predictions,
                                      target_names=['No Tumor', 'Tumor']))

        return results

    def save_ensemble(self, save_path='models/brain_tumor_ensemble.pkl'):
        """
        Save ensemble configuration for later use.

        Args:
            save_path (str): Path to save ensemble configuration
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ensemble_config = {
            'model_paths': self.model_paths,
            'weights': self.weights,
            'threshold': self.threshold,
            'ensemble_type': 'weighted_average'
        }

        with open(save_path, 'wb') as f:
            pickle.dump(ensemble_config, f)

        print(f"Ensemble configuration saved to: {save_path}")

    @classmethod
    def load_ensemble(cls, config_path='models/brain_tumor_ensemble.pkl'):
        """
        Load ensemble from saved configuration.

        Args:
            config_path (str): Path to ensemble configuration

        Returns:
            BrainTumorEnsemble: Loaded ensemble model
        """
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        ensemble = cls(
            model_paths=config['model_paths'],
            weights=config['weights'],
            threshold=config['threshold']
        )

        print(f"Ensemble loaded from: {config_path}")
        return ensemble


def load_validation_data():
    """Load and preprocess validation data."""
    val_data = np.load('processed_data/val_images.npz')
    val_images = val_data['images']
    val_labels = np.load('processed_data/val_labels.npy')

    if len(val_images.shape) == 3:
        val_images = np.stack([val_images]*3, axis=-1)

    return val_images, val_labels


def load_test_data():
    """Load and preprocess test data."""
    test_data = np.load('processed_data/test_images.npz')
    test_images = test_data['images']
    test_labels = np.load('processed_data/test_labels.npy')

    if len(test_images.shape) == 3:
        test_images = np.stack([test_images]*3, axis=-1)

    return test_images, test_labels


if __name__ == "__main__":
    # Create and test ensemble
    print("Creating Brain Tumor Detection Ensemble...")
    ensemble = BrainTumorEnsemble()

    # Load validation data
    print("\nLoading validation data...")
    val_images, val_labels = load_validation_data()

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = ensemble.evaluate(val_images, val_labels)

    # Load test data
    print("\nLoading test data...")
    test_images, test_labels = load_test_data()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = ensemble.evaluate(test_images, test_labels)

    # Save ensemble configuration
    print("\nSaving ensemble configuration...")
    ensemble.save_ensemble()

    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {val_results['accuracy']*100:.2f}%")
    print(f"Test Accuracy: {test_results['accuracy']*100:.2f}%")