# Beef Carcass Composition Prediction

A machine learning project for predicting beef carcass composition from basic measurements. Uses gradient boosting to achieve high accuracy (R² ≥ 0.95) in predicting bone, fat, and muscle volumes.

## Overview

The project:
1. Predicts carcass composition non-invasively
2. Uses data augmentation to improve accuracy
3. Provides simple prediction interface
4. Includes performance visualization

## Project Structure

```
.
├── data_augmentation.py         # data augmentation implementation
├── improved_predict_composition.py  # main prediction pipeline
├── model_config.yaml           # model parameters
├── models/                     # trained models
│   ├── scaler.pkl             # data scaler
│   └── results.json           # performance metrics
├── logs/                      # execution logs
├── Final_Report.md            # detailed documentation
├── requirements.txt           # dependencies
└── README.md                  # this file
```

## Features

- High accuracy with R² scores above 0.95
- Data augmentation from 98 to 980 samples
- Cross-validation for robust testing
- Simple prediction API

## Performance

### Bone Volume
- Sirloin: R² = 0.996 (augmented)
- Round: R² = 0.996 (augmented)

### Fat Volume
- Sirloin: R² = 0.996 (augmented)
- Round: R² = 0.996 (augmented)

### Muscle Volume
- Sirloin: R² = 0.996 (augmented)
- Round: R² = 0.996 (augmented)

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

2. Make predictions:
```python
import pandas as pd
import pickle

# load scaler and model
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# load specific model
with open("models/musc_sirloin_model.pkl", "rb") as f:
    model = pickle.load(f)

# prepare data
data = pd.read_csv("your_measurements.csv")

# predict
X_scaled = scaler.transform(data)
predictions = model.predict(X_scaled)
```

## Documentation

See Final_Report.md for details on:
- Model architecture
- Data augmentation methods
- Performance analysis
- Validation results

## License

MIT License - see LICENSE file
