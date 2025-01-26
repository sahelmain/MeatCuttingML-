import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import joblib

class BeefVolumePredictor:
    """Predicts tissue volumes for each cut using machine learning models."""

    def __init__(self):
        self.models: Dict[str, Dict[str, GradientBoostingRegressor]] = {
            'musc': {}, 'fat': {}, 'bone': {}
        }
        self.scalers: Dict[str, StandardScaler] = {}
        self.target_scalers: Dict[str, Dict[str, StandardScaler]] = {
            'musc': {}, 'fat': {}, 'bone': {}
        }
        self.feature_names: List[str] = []

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, NDArray[np.float64]]], Dict[str, Dict[str, NDArray[np.float64]]]]:
        # Define cuts and tissue types
        cuts = ['sirloin', 'round', 'loinwing', 'rib', 'plate', 'chuckroll', 'chuckclod', 'chuckbrisket']
        tissue_types = ['musc', 'fat', 'bone']

        # Prepare features (use non-volume measurements)
        feature_cols = [col for col in data.columns if not 'vol' in col and any(f'tot_{t}_' in col for t in tissue_types)]
        X = data[feature_cols]
        self.feature_names = feature_cols

        # Create train-test split indices
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)

        # Split and scale features
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        self.scalers['features'] = feature_scaler

        # Prepare target variables for each tissue type and cut
        y_train_dict = {tissue: {} for tissue in tissue_types}
        y_test_dict = {tissue: {} for tissue in tissue_types}

        for tissue in tissue_types:
            for cut in cuts:
                target_col = f'tot_{tissue}_vol_{cut}'
                if target_col in data.columns:
                    # Get target values
                    y = data[target_col]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]

                    # Scale target values
                    target_scaler = StandardScaler()
                    y_train_array = np.array(y_train.values, dtype=np.float64).reshape(-1, 1)
                    y_test_array = np.array(y_test.values, dtype=np.float64).reshape(-1, 1)
                    y_train_scaled = target_scaler.fit_transform(y_train_array)[:, 0]
                    y_test_scaled = target_scaler.transform(y_test_array)[:, 0]

                    # Store scaled values and scaler
                    y_train_dict[tissue][cut] = y_train_scaled
                    y_test_dict[tissue][cut] = y_test_scaled
                    self.target_scalers[tissue][cut] = target_scaler

        # Convert scaled features to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train_df, X_test_df, y_train_dict, y_test_dict

    def train_model(self, X_train: pd.DataFrame, y_train_dict: Dict[str, Dict[str, NDArray[np.float64]]]) -> None:
        for tissue in y_train_dict:
            print(f"\nTraining {tissue.upper()} models...")
            for cut in y_train_dict[tissue]:
                print(f"  Training {cut} model...")
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=5,
                    subsample=0.8,
                    random_state=42
                )
                model.fit(X_train, y_train_dict[tissue][cut])
                self.models[tissue][cut] = model

    def predict(self, X: pd.DataFrame, tissue_type: str, cut: str) -> NDArray[np.float64]:
        """Predict volume for a specific tissue type and cut."""
        if tissue_type not in self.models or cut not in self.models[tissue_type]:
            raise ValueError(f"No trained model found for tissue type: {tissue_type} and cut: {cut}")

        # Scale features
        X_scaled = self.scalers['features'].transform(X)

        # Get prediction
        y_pred_scaled = self.models[tissue_type][cut].predict(X_scaled)

        # Inverse transform prediction
        y_pred_array = np.array(y_pred_scaled, dtype=np.float64).reshape(-1, 1)
        y_pred = self.target_scalers[tissue_type][cut].inverse_transform(y_pred_array)[:, 0]

        return y_pred

    def evaluate(self, X_test: pd.DataFrame, y_test_dict: Dict[str, Dict[str, NDArray[np.float64]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        results = {}
        for tissue in y_test_dict:
            results[tissue] = {}
            print(f"\n{tissue.upper()} Results:")

            for cut in y_test_dict[tissue]:
                # Get predictions
                y_pred_scaled = self.models[tissue][cut].predict(X_test)
                y_true_scaled = y_test_dict[tissue][cut]

                # Calculate metrics on scaled data
                results[tissue][cut] = {
                    'r2_score': r2_score(y_true_scaled, y_pred_scaled),
                    'rmse': np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled)),
                    'mae': mean_absolute_error(y_true_scaled, y_pred_scaled)
                }

                print(f"\n  {cut.upper()}:")
                print(f"    R² Score: {results[tissue][cut]['r2_score']:.4f}")
                print(f"    RMSE: {results[tissue][cut]['rmse']:.4f}")
                print(f"    MAE: {results[tissue][cut]['mae']:.4f}")

                # Plot feature importance
                self.plot_feature_importance(tissue, cut)

        return results

    def plot_feature_importance(self, tissue: str, cut: str) -> None:
        model = self.models[tissue][cut]
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
        plt.title(f'Top 10 Important Features for {tissue.upper()} {cut.upper()} Volume Prediction')
        plt.tight_layout()

        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/{tissue}_{cut}_feature_importance.png')
        plt.close()

    def save_models(self) -> None:
        """Save models and scalers to their respective directories."""
        # Create directory structure
        os.makedirs('models/tissue_models', exist_ok=True)
        os.makedirs('models/scalers', exist_ok=True)

        # Save models
        for tissue in self.models:
            for cut in self.models[tissue]:
                joblib.dump(
                    self.models[tissue][cut],
                    f'models/tissue_models/{tissue}_{cut}_model.joblib'
                )

        # Save feature scaler
        joblib.dump(self.scalers['features'], 'models/scalers/feature_scaler.joblib')

        # Save target scalers
        for tissue in self.target_scalers:
            for cut in self.target_scalers[tissue]:
                joblib.dump(
                    self.target_scalers[tissue][cut],
                    f'models/scalers/{tissue}_{cut}_target_scaler.joblib'
                )

    @classmethod
    def load_models(cls, models_dir: str = 'models') -> 'BeefVolumePredictor':
        """Load a trained predictor from saved models and scalers."""
        predictor = cls()

        # Load feature scaler
        predictor.scalers['features'] = joblib.load(f'{models_dir}/scalers/feature_scaler.joblib')

        # Load models and target scalers
        for model_path in os.listdir(f'{models_dir}/tissue_models'):
            if model_path.endswith('_model.joblib'):
                # Parse tissue type and cut from filename
                parts = model_path.replace('_model.joblib', '').split('_')
                tissue = parts[0]
                cut = '_'.join(parts[1:])

                # Load model
                model = joblib.load(f'{models_dir}/tissue_models/{model_path}')
                predictor.models[tissue][cut] = model

                # Load corresponding target scaler
                scaler = joblib.load(f'{models_dir}/scalers/{tissue}_{cut}_target_scaler.joblib')
                predictor.target_scalers[tissue][cut] = scaler

        # Get feature names from the first model
        first_tissue = next(iter(predictor.models))
        first_cut = next(iter(predictor.models[first_tissue]))
        predictor.feature_names = predictor.models[first_tissue][first_cut].feature_names_in_.tolist()

        return predictor

def main():
    print("Loading and preparing data...")
    df = pd.read_csv('data/ct_composition.csv')

    # Create predictor instance
    predictor = BeefVolumePredictor()

    # Train and evaluate models
    X_train, X_test, y_train_dict, y_test_dict = predictor.prepare_data(df)
    predictor.train_model(X_train, y_train_dict)
    results = predictor.evaluate(X_test, y_test_dict)
    predictor.save_models()

    print("\nAnalysis complete! Check the 'outputs' directory for visualizations.")

if __name__ == "__main__":
    main()
