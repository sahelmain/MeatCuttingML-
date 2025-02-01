import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from data_augmentation import DataAugmenter
from typing import List, Dict, Optional, Union
import traceback
import sys
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BeefCompositionPredictor:
    """Predicts beef carcass composition from non-volume measurements."""

    def __init__(self,
                 beef_data_path: str = 'data/ct_composition.csv',
                 models_dir: str = 'models'):
        """Initialize the predictor with data paths and model parameters."""
        self.beef_data_path = Path(beef_data_path)
        self.models_dir = Path(models_dir)
        self.scaler = StandardScaler()
        self.best_models = {}
        self.results = {}  # Store training results

        # Define tissue types and cuts
        self.tissue_types = ['bone', 'fat', 'musc']
        self.cuts = ['sirloin', 'round', 'loinwing', 'rib', 'plate',
                    'chuckroll', 'chuckclod', 'chuckbrisket']

        # Optimized parameter grid for Gradient Boosting
        self.param_grid = {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 5, 6],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 0.9, 1.0]
        }

        # Initialize data augmenter
        self.augmenter = DataAugmenter()

    def load_and_preprocess_data(self):
        """
        Load and preprocess data, specifically separating:
        - Input (X): Non-volume measurements from ct_composition.csv
        - Output (y): Volume measurements for each tissue type and cut
        """
        # Load beef data
        beef_data = pd.read_csv(self.beef_data_path)
        beef_data['id'] = beef_data['id'].astype(int)
        logger.info(f"Loaded beef data: {len(beef_data)} samples")

        # Extract non-volume features as input
        input_features = [col for col in beef_data.columns
                         if not 'vol' in col.lower() and  # Exclude volume measurements
                         not 'unnamed' in col.lower() and # Exclude unnamed columns
                         col != 'id' and  # Exclude ID
                         col != '']  # Exclude empty column name

        logger.info(f"Using {len(input_features)} input features from ct_composition.csv")
        logger.info("Input features: " + ", ".join(input_features))

        # Prepare features (X): non-volume measurements
        X = beef_data[input_features]

        # Prepare targets (y): volume measurements
        y_dict = {}
        for tissue in self.tissue_types:
            y_dict[tissue] = {}
            for cut in self.cuts:
                col = f'tot_{tissue}_vol_{cut}'
                if col in beef_data.columns:
                    y_dict[tissue][cut] = beef_data[col]
                    logger.info(f"Target variable: {col} - {len(y_dict[tissue][cut])} measurements")

        return X, y_dict

    def _train_single_model(self, X: pd.DataFrame, y: pd.Series) -> tuple[GradientBoostingRegressor, float, dict]:
        """Train a single gradient boosting model with cross-validation."""
        param_grid = {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 0.9, 1.0]
        }

        model = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        r2 = grid_search.best_score_

        logger.info(f"Prediction accuracy (R²) = {r2:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # Store results
        return best_model, r2, grid_search.best_params_

    def train_models(self, X: pd.DataFrame, y_dict: Dict,
                    X_aug: Optional[pd.DataFrame] = None,
                    y_dict_aug: Optional[Dict] = None) -> None:
        """Train models to predict CT-derived volumes from non-volume measurements."""
        logger.info("\nTraining models on original and augmented data...")

        # First analyze muscle predictions
        tissue = 'musc'
        logger.info(f"\nTraining models to predict {tissue} volumes:")
        self.results[tissue] = {}

        for cut in y_dict[tissue].keys():
            logger.info(f"\nPredicting {tissue} volume in {cut} cut:")
            y = y_dict[tissue][cut]
            self.results[tissue][cut] = {}

            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(np.asarray(X_scaled), columns=X.columns, index=X.index)

            # Train model on original data
            logger.info(f"Using {X.shape[1]} measurements to predict volume")
            logger.info(f"Training on original dataset with {X.shape[0]} samples")

            model, r2, best_params = self._train_single_model(X_scaled, y)
            y_pred = model.predict(X_scaled)

            # Store results
            self.results[tissue][cut] = {
                'best_score': r2,
                'best_params': best_params,
                'data_source': 'original'
            }

            # Save model and create visualizations
            self.best_models[f"{tissue}_{cut}"] = {
                'model': model,
                'X': X_scaled,
                'y': y
            }
            self.plot_feature_importance(model, X.columns, tissue, cut)
            self.plot_predictions(y, y_pred, tissue, cut)

            # Train on augmented data if provided
            if X_aug is not None and y_dict_aug is not None:
                logger.info(f"Training on augmented dataset with {X_aug.shape[0]} samples")
                y_aug = y_dict_aug[tissue][cut]

                # Scale augmented features
                X_aug_scaled = self.scaler.transform(X_aug)
                X_aug_scaled = pd.DataFrame(np.asarray(X_aug_scaled), columns=X_aug.columns, index=X_aug.index)

                model_aug, r2_aug, best_params_aug = self._train_single_model(X_aug_scaled, y_aug)
                y_pred_aug = model_aug.predict(X_aug_scaled)

                # Store augmented results
                self.results[tissue][cut].update({
                    'best_score_augmented': r2_aug,
                    'best_params_augmented': best_params_aug,
                    'data_source': 'augmented'
                })

                # Create comparison plots
                self.plot_prediction_comparison(y, y_pred, y_aug, y_pred_aug, tissue, cut)
                self.plot_feature_importance_comparison(model, model_aug, X.columns, tissue, cut)

                # Save augmented model
                self.best_models[f"{tissue}_{cut}_augmented"] = {
                    'model': model_aug,
                    'X': X_aug_scaled,
                    'y': y_aug
                }

        # Then proceed with fat and bone predictions
        for tissue in ['fat', 'bone']:
            logger.info(f"\nTraining models to predict {tissue} volumes:")
            self.results[tissue] = {}

            for cut in y_dict[tissue].keys():
                logger.info(f"\nPredicting {tissue} volume in {cut} cut:")
                y = y_dict[tissue][cut]
                self.results[tissue][cut] = {}

                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                X_scaled = pd.DataFrame(np.asarray(X_scaled), columns=X.columns, index=X.index)

                # Train model on original data
                logger.info(f"Using {X.shape[1]} measurements to predict volume")
                logger.info(f"Training on original dataset with {X.shape[0]} samples")

                model, r2, best_params = self._train_single_model(X_scaled, y)
                y_pred = model.predict(X_scaled)

                # Store results
                self.results[tissue][cut] = {
                    'best_score': r2,
                    'best_params': best_params,
                    'data_source': 'original'
                }

                # Save model and create visualizations
                self.best_models[f"{tissue}_{cut}"] = {
                    'model': model,
                    'X': X_scaled,
                    'y': y
                }
                self.plot_feature_importance(model, X.columns, tissue, cut)
                self.plot_predictions(y, y_pred, tissue, cut)

                # Train on augmented data if provided
                if X_aug is not None and y_dict_aug is not None:
                    logger.info(f"Training on augmented dataset with {X_aug.shape[0]} samples")
                    y_aug = y_dict_aug[tissue][cut]

                    # Scale augmented features
                    X_aug_scaled = self.scaler.transform(X_aug)
                    X_aug_scaled = pd.DataFrame(np.asarray(X_aug_scaled), columns=X_aug.columns, index=X_aug.index)

                    model_aug, r2_aug, best_params_aug = self._train_single_model(X_aug_scaled, y_aug)
                    y_pred_aug = model_aug.predict(X_aug_scaled)

                    # Store augmented results
                    self.results[tissue][cut].update({
                        'best_score_augmented': r2_aug,
                        'best_params_augmented': best_params_aug,
                        'data_source': 'augmented'
                    })

                    # Create comparison plots
                    self.plot_prediction_comparison(y, y_pred, y_aug, y_pred_aug, tissue, cut)
                    self.plot_feature_importance_comparison(model, model_aug, X.columns, tissue, cut)

                    # Save augmented model
                    self.best_models[f"{tissue}_{cut}_augmented"] = {
                        'model': model_aug,
                        'X': X_aug_scaled,
                        'y': y_aug
                    }

    def plot_feature_importance(self, model, features, tissue, cut, is_augmented: bool = False):
        """Visualize which measurements are most predictive of volume."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 10  # Show top 10 most important features

        # Create more readable feature names
        feature_names = []
        for f in features:
            # Extract the measurement type and cut
            parts = f.split('_')
            if len(parts) >= 4:
                measure_type = parts[1]  # bone, fat, musc, wt
                measure_cut = parts[-1]  # cut name
                if measure_type == 'wt':
                    readable_name = f'Weight ({measure_cut})'
                else:
                    readable_name = f'{measure_type.capitalize()} ({measure_cut})'
                feature_names.append(readable_name)
            else:
                feature_names.append(f)

        plt.figure(figsize=(12, 6))
        title = f"Top {top_n} Most Predictive Measurements\n"
        title += f"for {tissue.capitalize()} Volume in {cut.capitalize()}"
        if is_augmented:
            title += " (Augmented Data)"
        plt.title(title)

        # Plot bars with different colors based on measurement type
        bars = plt.bar(range(top_n), importances[indices][:top_n])

        # Color code bars by measurement type
        colors = {
            'Weight': '#2ecc71',  # Green for weight
            'Bone': '#e74c3c',    # Red for bone
            'Fat': '#f1c40f',     # Yellow for fat
            'Musc': '#3498db'     # Blue for muscle
        }

        for idx, bar in enumerate(bars):
            feature_name = feature_names[indices[idx]]
            for measure_type, color in colors.items():
                if measure_type in feature_name:
                    bar.set_color(color)
                    break

        plt.xticks(range(top_n),
                  [feature_names[i] for i in indices][:top_n],
                  rotation=45, ha='right')
        plt.xlabel('Measurements')
        plt.ylabel('Predictive Importance')

        # Add legend
        legend_elements = [mpatches.Rectangle((0,0),1,1, facecolor=color, label=measure_type)
                         for measure_type, color in colors.items()]
        plt.legend(handles=legend_elements, title='Measurement Type',
                  loc='upper right', bbox_to_anchor=(1, 0.95))

        plt.tight_layout()
        suffix = "_augmented" if is_augmented else ""
        plt.savefig(f'outputs/plots/{tissue}_{cut}_feature_importance{suffix}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_predictions(self, y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray],
                        tissue_type: str = "", cut: str = "") -> None:
        """Visualize prediction accuracy with enhanced details."""
        y_pred = y_pred

        plt.figure(figsize=(10, 6))

        # Create scatter plot with density
        plt.hist2d(y_true, y_pred, bins=20, cmap='viridis')
        plt.colorbar(label='Number of Points')

        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                 label='Perfect Prediction', linewidth=2)

        # Calculate and display metrics
        r2 = r2_score(y_true, y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}'
        plt.text(0.05, 0.95, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')

        title = f'Prediction Accuracy: {tissue_type.capitalize()} Volume\n{cut.capitalize()} Cut'
        plt.title(title)
        plt.xlabel(f'Actual {tissue_type.capitalize()} Volume')
        plt.ylabel(f'Predicted {tissue_type.capitalize()} Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f'outputs/plots/{tissue_type}_{cut}_prediction_accuracy.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

    def save_results(self):
        """Save model results, performance metrics, and trained models."""
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Save detailed results
        detailed_results = {
            tissue: {
                cut: {
                    'prediction_accuracy': self.results[tissue][cut]['best_score'],
                    'model_parameters': self.results[tissue][cut]['best_params'],
                    'data_source': self.results[tissue][cut]['data_source'],
                    'predictive_measurements': {
                        str(feature): float(importance)
                        for feature, importance in zip(
                            self.best_models[f"{tissue}_{cut}"]['X'].columns,
                            self.best_models[f"{tissue}_{cut}"]['model'].feature_importances_
                        )
                    }
                }
                for cut in self.results[tissue]
            }
            for tissue in self.results
        }

        # Save results JSON
        with open(os.path.join(self.models_dir, "prediction_results.json"), 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Save scaler
        pd.to_pickle(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))

        # Save trained models
        for tissue in self.tissue_types:
            for cut in self.cuts:
                model_key = f"{tissue}_{cut}"
                if model_key in self.best_models:
                    model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.best_models[model_key]['model'], f)

                # Save augmented models if available
                aug_key = f"{model_key}_augmented"
                if aug_key in self.best_models:
                    aug_model_path = os.path.join(self.models_dir, f"{aug_key}_model.pkl")
                    with open(aug_model_path, 'wb') as f:
                        pickle.dump(self.best_models[aug_key]['model'], f)

        logger.info(f"Results, models, and scaler saved to {self.models_dir}")

    def plot_prediction_comparison(self, y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray],
                                 y_true_aug: Optional[pd.Series] = None,
                                 y_pred_aug: Optional[Union[pd.Series, np.ndarray]] = None,
                                 tissue_type: str = "", cut: str = "") -> None:
        """
        Create side-by-side scatter plots comparing original and augmented prediction accuracy.
        """
        # Convert numpy arrays to pandas Series if needed
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_true.index)
        if isinstance(y_pred_aug, np.ndarray):
            y_pred_aug = pd.Series(y_pred_aug, index=y_true_aug.index) if y_true_aug is not None else None

        plt.figure(figsize=(15, 6))

        # Original data subplot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        plt.title(f'Original Data (n=98)\nR² = {r2:.4f}, RMSE = {rmse:.2f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

        # Augmented data subplot (if provided)
        if y_true_aug is not None and y_pred_aug is not None:
            plt.subplot(1, 2, 2)
            plt.scatter(y_true_aug, y_pred_aug, alpha=0.5)
            min_val = min(y_true_aug.min(), y_pred_aug.min())
            max_val = max(y_true_aug.max(), y_pred_aug.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            r2_aug = r2_score(y_true_aug, y_pred_aug)
            rmse_aug = np.sqrt(mean_squared_error(y_true_aug, y_pred_aug))
            plt.title(f'Augmented Data (n=980)\nR² = {r2_aug:.4f}, RMSE = {rmse_aug:.2f}')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')

        plt.suptitle(f'{tissue_type.capitalize()} Volume Prediction - {cut.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'{tissue_type}_{cut}_prediction_comparison.png')
        plt.close()

    def plot_feature_importance_comparison(self, model: GradientBoostingRegressor,
                                         model_aug: Optional[GradientBoostingRegressor],
                                         feature_names: Union[List[str], pd.Index],
                                         tissue_type: str = "", cut: str = "") -> None:
        """
        Create side-by-side feature importance plots comparing original and augmented models.
        """
        feature_names_list = list(feature_names)  # Convert Index to List
        plt.figure(figsize=(15, 8))

        # Original model subplot
        plt.subplot(1, 2, 1)
        importance = pd.Series(model.feature_importances_, index=feature_names_list)
        importance = importance.sort_values(ascending=True)
        importance.plot(kind='barh')
        plt.title('Original Model\nFeature Importance')
        plt.xlabel('Importance Score')

        # Augmented model subplot
        if model_aug is not None:
            plt.subplot(1, 2, 2)
            importance_aug = pd.Series(model_aug.feature_importances_, index=feature_names_list)
            importance_aug = importance_aug.sort_values(ascending=True)
            importance_aug.plot(kind='barh')
            plt.title('Augmented Model\nFeature Importance')
            plt.xlabel('Importance Score')

        plt.suptitle(f'{tissue_type.capitalize()} Volume - {cut.capitalize()}\nFeature Importance Comparison')
        plt.tight_layout()
        plt.savefig(f'{tissue_type}_{cut}_feature_importance_comparison.png')
        plt.close()

def main():
    try:
        # Step 1: Load and preprocess original data
        logger.info("\nStep 1: Loading original data...")
        predictor = BeefCompositionPredictor()
        X, y_dict = predictor.load_and_preprocess_data()

        # Step 2: Augment data
        logger.info("\nStep 2: Augmenting data...")
        augmenter = DataAugmenter()
        X_aug, y_aug = augmenter.augment_data(X, y_dict)

        # Step 3: Train models and compare performance
        logger.info("\nStep 3: Training models and comparing performance...")
        predictor.train_models(X, y_dict, X_aug, y_aug)

        # Save results
        predictor.save_results()
        logger.info("\nPipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
