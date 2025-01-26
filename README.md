# Beef Carcass Composition Prediction

This project uses machine learning to predict beef carcass composition from live cattle morphology measurements using millimeter radar data.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── radar_measurements.csv    # Initial radar data (100 cattle)
│   ├── ct_composition.csv       # CT scan composition data
│   ├── augmented_radar.csv      # Augmented radar data (1000+ samples)
│   ├── wtamu_radar.csv         # WTAMU validation study radar data
│   └── wtamu_weights.csv       # WTAMU validation study weights
├── models/                # Saved models
├── outputs/              # Output files and visualizations
│   └── plots/            # Generated plots
├── scripts/              # Python scripts
│   └── predict_carcass_composition.py  # Main prediction script
├── config/               # Configuration files
└── README.md            # This file
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Project Phases

1. Initial Model Training
   - Train on data from 100 cattle
   - Uses CT scan data for ground truth
   - Evaluates model performance with cross-validation

2. Data Augmentation
   - Augment radar measurements to 1000+ samples
   - Integration with point cloud augmentation (Tommy Dang)

3. Model Application
   - Apply trained model to augmented data
   - Generate predictions for theoretical carcass composition

4. Validation
   - Validate model with West Texas A&M study data
   - Compare predictions with actual carcass component weights

## Usage

1. Activate the environment:
   ```bash
   conda activate beef_env
   ```

2. Run the prediction script:
   ```bash
   python scripts/predict_carcass_composition.py
   ```

3. Check results in the `outputs` directory:
   - Model performance metrics
   - Visualization plots
   - Predictions for augmented data
   - Validation results

## Model Performance

The model evaluates performance using:
- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Cross-validation Score

Results are saved in the outputs directory along with visualizations of:
- Actual vs Predicted plots
- Feature importance rankings
- Prediction error analysis
