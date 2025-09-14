"""
Deployment script for Brain Tumor Detection Ensemble
Production-ready inference interface for the 81.58% accuracy ensemble model.
"""

import numpy as np
from ensemble_model import BrainTumorEnsemble
import os
from PIL import Image
import argparse


class BrainTumorDetector:
    """
    Production-ready brain tumor detection interface.
    Uses the validated 81.58% accuracy ensemble model.
    """

    def __init__(self, config_path='models/brain_tumor_ensemble.pkl'):
        """
        Initialize detector with saved ensemble configuration.

        Args:
            config_path (str): Path to saved ensemble configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Ensemble configuration not found: {config_path}")

        self.ensemble = BrainTumorEnsemble.load_ensemble(config_path)
        print(f"Brain Tumor Detector loaded successfully")
        print(f"Expected accuracy: 81.58% on test set")

    def preprocess_image(self, image_path):
        """
        Preprocess single image for prediction.

        Args:
            image_path (str): Path to brain scan image

        Returns:
            numpy.ndarray: Preprocessed image array
        """
        img = Image.open(image_path)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((224, 224))  # Resize to model input size

        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.stack([img_array]*3, axis=-1)  # Convert to RGB
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array

    def predict_single(self, image_path, return_confidence=True):
        """
        Predict tumor presence in a single brain scan.

        Args:
            image_path (str): Path to brain scan image
            return_confidence (bool): Return confidence score

        Returns:
            dict: Prediction results
        """
        img_array = self.preprocess_image(image_path)

        # Get probability
        probability = self.ensemble.predict(img_array, return_probabilities=True)[0][0]

        # Get binary prediction
        prediction = int(probability > self.ensemble.threshold)

        result = {
            'prediction': prediction,
            'label': 'Tumor' if prediction == 1 else 'No Tumor',
            'confidence': float(probability),
            'threshold': self.ensemble.threshold
        }

        if return_confidence:
            confidence_level = 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'
            result['confidence_level'] = confidence_level

        return result

    def predict_batch(self, image_paths):
        """
        Predict tumor presence in multiple brain scans.

        Args:
            image_paths (list): List of paths to brain scan images

        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                })

        return results


def validate_deployment():
    """Validate deployment with test data."""
    print("Validating deployment with test data...")

    try:
        # Load ensemble
        detector = BrainTumorDetector()

        # Load test data for validation
        from ensemble_model import load_test_data
        test_images, test_labels = load_test_data()

        # Make predictions
        correct = 0
        total = len(test_labels)

        for i in range(total):
            # Save temporary image for testing
            temp_path = f"temp_test_image_{i}.png"
            img = Image.fromarray((test_images[i, :, :, 0] * 255).astype(np.uint8))
            img.save(temp_path)

            # Predict
            result = detector.predict_single(temp_path, return_confidence=False)
            if result['prediction'] == test_labels[i]:
                correct += 1

            # Clean up
            os.remove(temp_path)

        accuracy = correct / total
        print(f"Deployment validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        if accuracy >= 0.81:
            print("Deployment validation: SUCCESS")
            return True
        else:
            print("Deployment validation: FAILED")
            return False

    except Exception as e:
        print(f"Deployment validation error: {e}")
        return False


def main():
    """Main CLI interface for brain tumor detection."""
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Ensemble')
    parser.add_argument('--image', type=str, help='Path to brain scan image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple brain scan images')
    parser.add_argument('--validate', action='store_true', help='Validate deployment')
    parser.add_argument('--config', type=str, default='models/brain_tumor_ensemble.pkl',
                       help='Path to ensemble configuration')

    args = parser.parse_args()

    if args.validate:
        success = validate_deployment()
        exit(0 if success else 1)

    if not (args.image or args.batch):
        print("Please provide --image, --batch, or --validate option")
        parser.print_help()
        return

    # Initialize detector
    detector = BrainTumorDetector(args.config)

    if args.image:
        # Single image prediction
        print(f"Analyzing: {args.image}")
        result = detector.predict_single(args.image)

        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Confidence Level: {result['confidence_level']}")
        print(f"Threshold: {result['threshold']}")

    elif args.batch:
        # Batch prediction
        print(f"Analyzing {len(args.batch)} images...")
        results = detector.predict_batch(args.batch)

        for result in results:
            if result['status'] == 'success':
                print(f"{result['image_path']}: {result['label']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"{result['image_path']}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()