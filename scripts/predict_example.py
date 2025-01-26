import pandas as pd
import numpy as np
from predict_volumes import BeefVolumePredictor

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction by filtering volume-related columns."""
    tissue_types = ['musc', 'fat', 'bone']
    # Keep only non-volume measurements
    feature_cols = [col for col in data.columns
                   if not 'vol' in col and any(f'tot_{t}_' in col for t in tissue_types)]
    return data[feature_cols]

def main():
    # Load the trained predictor
    print("Loading trained models...")
    predictor = BeefVolumePredictor.load_models()

    # Load some example data
    print("\nLoading example data...")
    example_data = pd.read_csv('data/ct_composition.csv')

    # Select a single sample for demonstration
    sample = example_data.iloc[[0]]

    # Prepare features
    features = prepare_features(sample)

    # Make predictions for each tissue type and cut
    tissue_types = ['musc', 'fat', 'bone']
    cuts = ['sirloin', 'round', 'loinwing', 'rib', 'plate', 'chuckroll', 'chuckclod', 'chuckbrisket']

    print("\nMaking predictions:")
    for tissue in tissue_types:
        print(f"\n{tissue.upper()} Predictions:")
        for cut in cuts:
            try:
                # Make prediction
                pred_volume = predictor.predict(features, tissue, cut)

                # Get actual volume for comparison
                actual_volume = sample[f'tot_{tissue}_vol_{cut}'].values[0]

                # Print results
                print(f"\n  {cut.upper()}:")
                print(f"    Predicted Volume: {pred_volume[0]:.2f}")
                print(f"    Actual Volume:    {actual_volume:.2f}")
                print(f"    Difference:       {abs(pred_volume[0] - actual_volume):.2f}")
                print(f"    % Error:          {abs(pred_volume[0] - actual_volume) / actual_volume * 100:.2f}%")
            except KeyError:
                print(f"  {cut.upper()}: No volume data available")
            except Exception as e:
                print(f"  {cut.upper()}: Error making prediction - {str(e)}")

if __name__ == "__main__":
    main()
