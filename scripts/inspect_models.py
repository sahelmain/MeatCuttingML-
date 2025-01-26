import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict

def load_and_inspect_models() -> None:
    # Load models for each tissue type
    print("Loading models...")
    for tissue in ['musc', 'fat', 'bone']:
        model = joblib.load(f'models/{tissue}_model.joblib')
        print(f"\n{tissue.upper()} Model Details:")
        print("Model Type:", type(model).__name__)
        print("Feature Importances:")

        # Get feature names if available
        try:
            feature_names = model.feature_names_in_
            importances = model.feature_importances_

            # Create a dataframe of feature importances
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # Display top 10 most important features
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))

        except AttributeError:
            print("Feature names not available in model")

        print("\nModel Parameters:")
        for param, value in model.get_params().items():
            print(f"{param}: {value}")

        print("-" * 80)

    # Load and inspect scaler
    print("\nLoading Scaler...")
    try:
        scaler = joblib.load('models/X_scaler.joblib')
        print("Scaler Type:", type(scaler).__name__)
        print("Scale mean:", scaler.mean_)
        print("Scale std:", scaler.scale_)
    except FileNotFoundError:
        print("Scaler file not found")

if __name__ == "__main__":
    load_and_inspect_models()
