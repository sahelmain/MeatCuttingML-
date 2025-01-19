# Beef Cut Volume Predictor

A machine learning model that predicts sirloin volume based on various beef cut weights.

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── requirements.txt   # Python dependencies
│   └── setup.py          # Package setup configuration
│
├── data/                  # Data directory
│   └── beef_data.csv     # Beef cut analysis data
│
├── scripts/              # Analysis and utility scripts
│   └── analyze_beef_data.py  # Main analysis and model script
│
├── src/                  # Source code
│   ├── models/          # Model implementations
│   └── utils/           # Utility functions
│
└── outputs/             # Model outputs (created during runtime)
    ├── best_model.pth   # Best model checkpoint
    ├── X_scaler.joblib  # Feature scaler
    └── y_scaler.joblib  # Target scaler
```

## Setup

1. Install dependencies:
```bash
pip install -r config/requirements.txt
```

2. Install package:
```bash
pip install -e .
```

## Usage

Run the analysis and train the model:
```bash
python scripts/analyze_beef_data.py
```

## Model Architecture

The project uses an ensemble of neural networks with:
- Multi-head attention mechanism
- Feature interaction layers
- Advanced residual blocks
- Ensemble prediction capability

## Performance

- R² Score: 0.9742 (97.42% accuracy)
- Root Mean Squared Error: 552.53
- Mean Squared Error: 305,289.45

## Data Fields Definition

### Input Variables
1. `tot_wt_chuckbrisket`: Total weight of chuck brisket cut (in pounds)
2. `tot_wt_chuckclod`: Total weight of chuck clod cut (in pounds)
3. `tot_wt_chuckroll`: Total weight of chuck roll cut (in pounds)
4. `tot_wt_loinwing`: Total weight of loin wing cut (in pounds)
5. `tot_wt_plate`: Total weight of plate cut (in pounds)
6. `tot_wt_rib`: Total weight of rib cut (in pounds)
7. `tot_wt_round`: Total weight of round cut (in pounds)
8. `tot_wt_sirloin`: Total weight of sirloin cut (in pounds)

### Output Variable
- `tot_vol_sirloin`: Total volume of sirloin (in cubic inches)

### Future Integration Points
- Radar point cloud data (pending from Garrett)
- ID system to link radar data with weight measurements
- Additional measurement correlations 