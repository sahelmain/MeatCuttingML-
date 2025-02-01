# Beef Carcass Composition Prediction

This project uses machine learning to predict beef carcass composition (bone, fat, and muscle volumes) from non-volume measurements. The models achieve high accuracy (R² ≥ 0.95) through gradient boosting and data augmentation techniques.

## Project Overview

The project aims to:
1. Predict carcass composition without invasive procedures
2. Improve prediction accuracy through data augmentation
3. Provide an easy-to-use interface for making predictions
4. Generate comprehensive performance visualizations

## Project Structure

```
.
├── data_augmentation.py         # Data augmentation implementation
├── improved_predict_composition.py  # Main prediction pipeline
├── model_config.yaml           # Model configuration parameters
├── models/                     # Trained models and results
│   ├── scaler.pkl             # Feature scaler
│   └── results.json           # Performance metrics
├── logs/                      # Execution logs
├── Final_Report.md            # Comprehensive project report
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Key Features

- **High Accuracy**: Achieves R² scores ≥ 0.95 for all tissue types
- **Data Augmentation**: Expands training data from 98 to 980 samples
- **Robust Validation**: Cross-validation and performance visualization
- **Easy Integration**: Simple API for making predictions

## Model Performance

### Bone Volume Predictions
- **Sirloin**: R² = 0.996 (Augmented)
- **Round**: R² = 0.996 (Augmented)

### Fat Volume Predictions
- **Sirloin**: R² = 0.996 (Augmented)
- **Round**: R² = 0.996 (Augmented)

### Muscle Volume Predictions
- **Sirloin**: R² = 0.996 (Augmented)
- **Round**: R² = 0.996 (Augmented)

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run predictions:
```python
import pandas as pd
import pickle

# Load the scaler and model
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load specific model (e.g., muscle in sirloin)
with open("models/musc_sirloin_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare your data
data = pd.read_csv("your_measurements.csv")

# Scale and predict
X_scaled = scaler.transform(data)
predictions = model.predict(X_scaled)
```

## Documentation

For detailed information about:
- Model architecture and training
- Data augmentation methodology
- Performance analysis
- Validation results

Please refer to [Final_Report.md](Final_Report.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
