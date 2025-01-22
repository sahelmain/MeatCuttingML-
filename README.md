# Beef Cut Volume Predictor

A machine learning model that predicts sirloin volume based on various beef cut weights, CT scans, and millimeter radar measurements.

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── requirements.txt   # Python dependencies
│   └── setup.py          # Package setup configuration
│
├── data/                  # Data directory
│   ├── beef_data.csv     # Beef cut analysis data
│   ├── ct_scans/         # CT scan data
│   └── radar_data/       # Millimeter radar measurements
│
├── matlab/               # MATLAB scripts
│   ├── scripts/
│   │   ├── process_ct_data.m      # CT data processing
│   │   ├── process_radar_data.m   # Radar data processing
│   │   ├── data_fusion.m          # Multi-modal data fusion
│   │   └── data_augmentation.m    # Data augmentation
│   └── functions/        # Helper functions
│
├── scripts/              # Python analysis scripts
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

3. MATLAB Requirements:
- MATLAB R2023a or later
- Image Processing Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

## Usage

1. Process CT and Radar Data:
```matlab
% In MATLAB
cd matlab/scripts
ct_features = process_ct_data('path/to/ct/scan.dcm');
radar_features = process_radar_data('path/to/radar/data.mat');
fused_features = data_fusion(ct_features, radar_features);
```

2. Run the analysis and train the model:
```bash
python scripts/analyze_beef_data.py
```

## Model Architecture

The project uses a multi-modal approach:
1. CT Scan Processing:
   - Hounsfield unit normalization
   - Tissue segmentation
   - Volume calculation
   - Shape feature extraction

2. Millimeter Radar Processing:
   - Signal processing
   - Range profile analysis
   - Doppler processing
   - Surface feature extraction

3. Data Fusion:
   - Feature-level fusion
   - Confidence-weighted combination
   - Multi-modal validation

4. Neural Network Ensemble:
   - Multi-head attention mechanism
   - Feature interaction layers
   - Advanced residual blocks
   - Ensemble prediction capability

## Performance

- R² Score: 0.9560 (95.60% accuracy)
- Root Mean Squared Error: 722.05
- Mean Squared Error: 521,362.16

## Data Fields Definition

### Input Variables
1. CT Scan Features:
   - Tissue composition ratios
   - Volume measurements
   - Shape metrics

2. Radar Features:
   - Surface morphology
   - Motion characteristics
   - Surface metrics

3. Weight Measurements:
   - `tot_wt_chuckbrisket`: Total weight of chuck brisket cut (in pounds)
   - `tot_wt_chuckclod`: Total weight of chuck clod cut (in pounds)
   - `tot_wt_chuckroll`: Total weight of chuck roll cut (in pounds)
   - `tot_wt_loinwing`: Total weight of loin wing cut (in pounds)
   - `tot_wt_plate`: Total weight of plate cut (in pounds)
   - `tot_wt_rib`: Total weight of rib cut (in pounds)
   - `tot_wt_round`: Total weight of round cut (in pounds)
   - `tot_wt_sirloin`: Total weight of sirloin cut (in pounds)

### Output Variable
- `tot_vol_sirloin`: Total volume of sirloin (in cubic inches)

### Data Augmentation
- CT scan variations
- Radar measurement augmentation
- Synthetic data generation
- Cross-modal consistency enforcement

## Future Integration Points
- Enhanced radar point cloud processing
- Advanced CT-radar fusion techniques
- Real-time processing capabilities
- Automated calibration system
