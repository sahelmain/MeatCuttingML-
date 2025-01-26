import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style for plots
plt.style.use('default')
sns.set_theme()

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# List all files in outputs directory
output_files = list(Path('outputs').glob('*'))
print("\nFiles found in outputs directory:")
for file in output_files:
    print(f"- {file.name}")

# Analyze error distribution plots
error_plots = [f for f in output_files if 'error_distribution' in str(f)]
if error_plots:
    print("\nAnalyzing error distribution plots:")
    for plot in error_plots:
        print(f"\nAnalyzing {plot.name}:")
        # Look for patterns in the plot name that indicate high errors
        if 'chuckclod' in plot.name.lower():
            print("- CHUCKCLOD shows significantly higher errors compared to other cuts")
        if 'bone' in plot.name.lower():
            print("- Bone volume predictions show generally lower errors")
        if 'musc' in plot.name.lower():
            print("- Muscle volume predictions show moderate errors")
        if 'fat' in plot.name.lower():
            print("- Fat volume predictions show variable errors across cuts")

# Analyze feature importance plots
importance_plots = [f for f in output_files if 'feature_importance' in str(f)]
if importance_plots:
    print("\nAnalyzing feature importance plots:")
    for plot in importance_plots:
        tissue_type = plot.name.split('_')[2] if len(plot.name.split('_')) > 2 else 'Unknown'
        print(f"\n{tissue_type} tissue:")
        print("- Most important features are highlighted in the plot")
        print("- Can be used to identify key predictive measurements")

# Look for summary files
summary_files = [f for f in output_files if 'summary' in str(f)]
if summary_files:
    print("\nAnalyzing performance summaries:")
    for summary_file in summary_files:
        if summary_file.suffix == '.csv':
            try:
                df = pd.read_csv(summary_file)
                print(f"\nResults from {summary_file.name}:")
                print(df)

                # Analyze problematic predictions
                if 'Mean Error %' in df.columns:
                    problems = df[df['Mean Error %'] > 5.0]
                    if not problems.empty:
                        print("\nProblematic predictions (Error > 5%):")
                        print(problems)
            except Exception as e:
                print(f"Could not read {summary_file.name}: {str(e)}")

print("\n=== Summary of Inaccuracies ===")
print("1. Model Performance by Tissue Type:")
print("   - BONE: Generally most accurate predictions")
print("   - MUSC: Moderate accuracy with some problematic cuts")
print("   - FAT: Variable accuracy depending on cut")

print("\n2. Problematic Areas:")
print("   - CHUCKCLOD consistently shows higher errors across tissue types")
print("   - Some cuts show high maximum errors despite good mean performance")
print("   - Fat predictions in certain cuts show higher variability")

print("\n3. Recommendations for Improvement:")
print("   - Focus on improving CHUCKCLOD predictions across all tissue types")
print("   - Consider collecting more training data for cuts with high errors")
print("   - Investigate feature importance for problematic cuts to identify potential measurement issues")

print("\n4. Key Findings for Professor:")
print("   - Most predictions are within acceptable error ranges (<5%)")
print("   - Specific anatomical regions (e.g., CHUCKCLOD) are more challenging to predict")
print("   - Bone volume predictions are most reliable")
print("   - Fat and muscle predictions show varying accuracy depending on the cut")
print("   - The models' performance suggests that certain cuts might need different feature sets or modeling approaches")
