import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any
import os
import joblib

class CarcassCompositionPredictor:
    """Predicts carcass composition from radar measurements."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.feature_importances = None

    def prepare_data(self, radar_data: pd.DataFrame, ct_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Prepare radar and CT data for training or prediction."""
        # Process radar point cloud data
        processed_radar = self._process_radar_features(radar_data)
        return processed_radar, ct_data

    def _process_radar_features(self, radar_data: pd.DataFrame) -> pd.DataFrame:
        """Process radar point cloud data into meaningful features."""
        # TODO: Implement radar point cloud processing
        # This will be integrated with Tommy Dang's data augmentation work
        return radar_data

    def train_model(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[Dict[str, float], Tuple]:
        """Train the model and return performance metrics."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Calculate feature importance
        self.feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'cv_score': np.mean(cross_val_score(self.model, X_scaled, y, cv=5))
        }

        return metrics, (X_test, y_test, y_pred)

    def predict(self, radar_data: pd.DataFrame) -> np.ndarray:
        """Predict carcass composition from radar data."""
        X, _ = self.prepare_data(radar_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def apply_to_augmented_data(self, augmented_data: pd.DataFrame) -> pd.DataFrame:
        """Apply model to augmented radar data."""
        predictions = self.predict(augmented_data)
        return pd.DataFrame(predictions, columns=['predicted_composition'])

    def validate_with_wtamu_data(self, radar_data: pd.DataFrame, actual_weights: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray]:
        """Validate model with West Texas A&M data."""
        predictions = self.predict(radar_data)

        # Calculate validation metrics
        metrics = {
            'r2': r2_score(actual_weights, predictions),
            'rmse': np.sqrt(mean_squared_error(actual_weights, predictions)),
            'mae': mean_absolute_error(actual_weights, predictions)
        }

        return metrics, predictions

    def plot_results(self, results: Tuple, plot_type: str = 'training'):
        """Generate and save visualization plots."""
        X_test, y_test, y_pred = results

        # Create plots directory
        os.makedirs('outputs/plots', exist_ok=True)

        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Composition')
        plt.ylabel('Predicted Composition')
        plt.title(f'Carcass Composition: Actual vs Predicted ({plot_type})')
        plt.tight_layout()
        plt.savefig(f'outputs/plots/prediction_{plot_type}.png')
        plt.close()

        if plot_type == 'training' and self.feature_importances is not None:
            # Feature Importance plot
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature',
                       data=self.feature_importances.head(10))
            plt.title('Top 10 Important Radar Features')
            plt.tight_layout()
            plt.savefig('outputs/plots/feature_importance.png')
            plt.close()

    def save_model(self):
        """Save trained model and scaler."""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/carcass_composition_model.joblib')
        joblib.dump(self.scaler, 'models/carcass_composition_scaler.joblib')

def main():
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    # 1. Load and prepare initial training data (100 cattle)
    print("Loading training data...")
    radar_data = pd.read_csv('data/radar_measurements.csv')  # TODO: Update with actual file
    ct_data = pd.read_csv('data/ct_composition.csv')        # TODO: Update with actual file

    # Initialize predictor
    predictor = CarcassCompositionPredictor()

    # 2. Train initial model
    print("\nTraining model on initial data...")
    X, y = predictor.prepare_data(radar_data, ct_data)
    if y is None:
        raise ValueError("CT data is required for training")
    metrics, results = predictor.train_model(X, y)

    # Print training metrics
    print("\nTraining Results:")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"Cross-validation Score: {metrics['cv_score']:.4f}")

    # Plot training results
    predictor.plot_results(results, 'training')

    # 3. Apply to augmented data (1000+ samples)
    print("\nProcessing augmented data...")
    augmented_data = pd.read_csv('data/augmented_radar.csv')  # TODO: Update with actual file
    augmented_predictions = predictor.apply_to_augmented_data(augmented_data)

    # Save augmented predictions
    augmented_predictions.to_csv('outputs/augmented_predictions.csv', index=False)

    # 4. Validate with West Texas A&M data
    print("\nValidating with WTAMU data...")
    wtamu_radar = pd.read_csv('data/wtamu_radar.csv')        # TODO: Update with actual file
    wtamu_weights = pd.read_csv('data/wtamu_weights.csv')    # TODO: Update with actual file

    validation_metrics, validation_predictions = predictor.validate_with_wtamu_data(
        wtamu_radar, wtamu_weights
    )

    # Print validation metrics
    print("\nValidation Results:")
    print(f"R² Score: {validation_metrics['r2']:.4f}")
    print(f"RMSE: {validation_metrics['rmse']:.2f}")
    print(f"MAE: {validation_metrics['mae']:.2f}")

    # Save model
    predictor.save_model()

    print("\nAnalysis complete! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()
