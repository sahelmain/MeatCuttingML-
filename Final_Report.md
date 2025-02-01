# Beef Carcass Composition Prediction: Final Report

## Table of Contents
1. Introduction
2. Data and Methods
3. Model Training and Augmentation
4. Results and Discussion
5. Conclusions and Future Directions
6. References

## 1. Introduction
Predicting beef carcass composition without fully invasive procedures can save significant time and resources in the beef industry. In this project, I developed machine learning models that estimate **bone**, **fat**, and **muscle volumes** using only **non-volume measurements** (e.g., total bone, total fat, total muscle, weight).

### Key Steps in the Project
- **Data Gathering:** CT scans of carcasses served as the main data source.
- **Data Augmentation:** Expanded the dataset via controlled noise injection.
- **Machine Learning Modeling:** Employed gradient boosting regression.
- **Evaluation and Visualization:** Measured accuracy (R², RMSE) and generated plots of predictions and feature importance.

### Project Goals
1. **Build** gradient boosting regressors using basic anatomical measurements as inputs.
2. **Augment** the limited original dataset to improve generalization.
3. **Compare** performance between original and augmented datasets.
4. **Visualize** feature importance and model predictions clearly.

## 2. Data and Methods

### 2.1 Data Source
- **ct_composition.csv**
  - Non-volume measurements for each beef cut (bone, fat, muscle, weight).
  - Volume measurements (targets) for bone, fat, and muscle in each cut.

### 2.2 Features and Targets
- **Features (Inputs):**
  - Columns such as tot_bone_*, tot_fat_*, tot_musc_*, and tot_wt_* (excluding _vol_).
- **Targets (Outputs):**
  - Columns labeled tot_bone_vol_*, tot_fat_vol_*, and tot_musc_vol_*.

### 2.3 Data Augmentation
To address the small initial dataset (98 samples), I implemented **data augmentation**:

1. **Noise Injection:** Added Gaussian noise proportional to each feature's standard deviation.
2. **Statistical Checks:** Ensured augmented samples maintained similar distributions and correlations.

This increased the training set from 98 to 980 samples.

## 3. Model Training and Augmentation

### 3.1 Model Choice and Hyperparameters
I used a **Gradient Boosting Regressor** optimized via **GridSearchCV**. Key hyperparameters included:
- **n_estimators:** [200, 300, 400]
- **learning_rate:** [0.01, 0.05, 0.1]
- **max_depth:** [3, 4, 5]
- **min_samples_split:** [2, 5]
- **subsample:** [0.8, 0.9, 1.0]

### 3.2 Workflow Overview
1. **Load Data:** Read ct_composition.csv; separate features (X) from volume targets (y).
2. **Augment Data:** Inject noise to create synthetic samples, expanding the dataset.
3. **Train Models:** Use original and augmented data to train gradient boosting regressors.
4. **Evaluate Performance:** Track R² and RMSE.
5. **Visualize Results:** Generate plots comparing predictions vs. actual values, along with feature importance.

## 4. Results and Discussion

### 4.1 Bone Volume
- **Sirloin**
  - Original: R² ≈ 0.977
  - Augmented: R² ≈ 0.996

- **Round**
  - Original: R² ≈ 0.981
  - Augmented: R² ≈ 0.996

*Interpretation:* Even with initially high accuracy, data augmentation boosted performance to above 0.99 for both Sirloin and Round. Deeper trees (max_depth=5) and partial subsampling helped avoid overfitting.

### 4.2 Fat Volume
- **Sirloin**
  - Original: R² ≈ 0.993
  - Augmented: R² ≈ 0.996

- **Round**
  - Original: R² ≈ 0.988
  - Augmented: R² ≈ 0.996

*Interpretation:* Fat volume was already easier to predict (≥ 0.98 R²). Augmentation still yielded modest gains up to ~0.996. Strong correlations between certain weight measurements and fat may explain the consistently high base accuracy.

### 4.3 Muscle Volume
- **Sirloin**
  - Original: R² ≈ 0.982
  - Augmented: R² ≈ 0.996

- **Round**
  - Original: R² ≈ 0.925
  - Augmented: R² ≈ 0.996

*Interpretation:* Muscle volume posed a greater challenge (especially for Round). Data augmentation boosted Round muscle R² by over 7 percentage points, indicating the original dataset lacked sufficient variation for robust muscle estimates.

### 4.4 Overall Observations
1. **Consistent Improvements:** All tissue types showed higher R² post-augmentation.
2. **Enhanced Feature Importance:** Weight and adjacent-cut metrics (e.g., bone in neighboring cuts) gained importance after augmentation.
3. **Model Complexity:** Augmented models often used deeper trees and lower learning rates without overfitting, thanks to the bigger synthetic dataset.

## 5. Conclusions and Future Directions

1. **High Accuracy:** Predicting carcass composition from non-invasive measurements can yield R² ≥ 0.95, particularly with augmented data.
2. **Augmentation Impact:** Greatest gains observed in muscle predictions (especially challenging cuts).
3. **Industry Potential:** Offers a pathway to faster, less-invasive carcass evaluations.
4. **Future Steps:**
   - Collect more real-world data for external validation.
   - Explore advanced augmentation (e.g., generative models).
   - Investigate integrating other data sources (e.g., millimeter-wave radar for live cattle morphology).

## 6. References
1. Geissler Corporation. *SizeR©* Millimeter Radar System. Minneapolis, MN.
2. Washington State University Veterinary Teaching Hospital. *CT Carcass Scanning Protocol.* Pullman, WA.

## Appendix: Using the Trained Models

1. **Load** the scaler (scaler.pkl) and the model files (e.g., musc_sirloin_model.pkl).
2. **Prepare** new data matching the original non-volume columns (e.g., tot_bone_*, tot_fat_*, tot_musc_*, tot_wt_*).
3. **Scale and Predict** using the loaded scaler and model.

**Quick Example (Python)**
```python
import pandas as pd
import pickle

# Load the scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load a specific model, e.g., muscle in the sirloin cut
with open("models/musc_sirloin_model.pkl", "rb") as f:
    musc_sirloin_model = pickle.load(f)

# Prepare new data
new_data = pd.read_csv("my_new_measurements.csv")

# Scale and predict
X_scaled = scaler.transform(new_data)
predictions = musc_sirloin_model.predict(X_scaled)
print("Predicted Sirloin Muscle Volumes:", predictions)
```

**Note:** Add generated plots from the outputs/plots directory to illustrate these results.
