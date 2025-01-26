import pandas as pd
import numpy as np
from predict_volumes import BeefVolumePredictor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction by filtering volume-related columns."""
    tissue_types = ['musc', 'fat', 'bone']
    feature_cols = [col for col in data.columns
                   if not 'vol' in col and any(f'tot_{t}_' in col for t in tissue_types)]
    return data[feature_cols]

def evaluate_predictions(predictor: BeefVolumePredictor,
                       data: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate model predictions on the entire dataset."""
    features = prepare_features(data)

    # Initialize results storage
    all_results = {
        'musc': {}, 'fat': {}, 'bone': {}
    }
    predictions_df = []

    tissue_types = ['musc', 'fat', 'bone']
    cuts = ['sirloin', 'round', 'loinwing', 'rib', 'plate', 'chuckroll', 'chuckclod', 'chuckbrisket']

    for tissue in tissue_types:
        all_results[tissue] = {}
        for cut in cuts:
            try:
                # Make predictions
                pred_volumes = predictor.predict(features, tissue, cut)
                actual_volumes = data[f'tot_{tissue}_vol_{cut}'].values

                # Calculate metrics
                abs_errors = np.abs(pred_volumes - actual_volumes)
                rel_errors = abs_errors / actual_volumes * 100

                metrics = {
                    'mean_abs_error': np.mean(abs_errors),
                    'median_abs_error': np.median(abs_errors),
                    'mean_rel_error': np.mean(rel_errors),
                    'median_rel_error': np.median(rel_errors),
                    'max_rel_error': np.max(rel_errors),
                    'min_rel_error': np.min(rel_errors),
                    'std_rel_error': np.std(rel_errors)
                }

                all_results[tissue][cut] = metrics

                # Store predictions for plotting
                for i in range(len(pred_volumes)):
                    predictions_df.append({
                        'Tissue': tissue,
                        'Cut': cut,
                        'Predicted': pred_volumes[i],
                        'Actual': actual_volumes[i],
                        'Error': rel_errors[i]
                    })

            except Exception as e:
                print(f"Error processing {tissue} {cut}: {str(e)}")

    return all_results, pd.DataFrame(predictions_df)

def plot_error_distributions(predictions_df: pd.DataFrame, output_dir: str):
    """Plot error distributions for each tissue type and cut."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Overall error distribution by tissue type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=predictions_df, x='Tissue', y='Error')
    plt.title('Prediction Error Distribution by Tissue Type')
    plt.ylabel('Relative Error (%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_by_tissue.png')
    plt.close()

    # Error distribution by cut for each tissue type
    for tissue in predictions_df['Tissue'].unique():
        plt.figure(figsize=(15, 6))
        tissue_data = predictions_df[predictions_df['Tissue'] == tissue]
        sns.boxplot(data=tissue_data, x='Cut', y='Error')
        plt.title(f'{tissue.upper()} Prediction Error Distribution by Cut')
        plt.ylabel('Relative Error (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_{tissue}_by_cut.png')
        plt.close()

def plot_prediction_scatter(predictions_df: pd.DataFrame, output_dir: str):
    """Create scatter plots of predicted vs actual values."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Scatter plot for each tissue type
    for tissue in predictions_df['Tissue'].unique():
        plt.figure(figsize=(10, 10))
        tissue_data = predictions_df[predictions_df['Tissue'] == tissue]

        plt.scatter(tissue_data['Actual'], tissue_data['Predicted'], alpha=0.5)

        # Add perfect prediction line
        min_val = min(tissue_data['Actual'].min(), tissue_data['Predicted'].min())
        max_val = max(tissue_data['Actual'].max(), tissue_data['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.title(f'{tissue.upper()} Volume Predictions')
        plt.xlabel('Actual Volume')
        plt.ylabel('Predicted Volume')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scatter_{tissue}.png')
        plt.close()

def generate_summary_report(results: Dict, predictions_df: pd.DataFrame, output_dir: str):
    """Generate a markdown report summarizing model performance."""
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/model_performance.md', 'w') as f:
        f.write('# Beef Carcass Volume Prediction Model Performance\n\n')

        # Overall statistics
        f.write('## Overall Performance Summary\n\n')
        for tissue in results:
            tissue_errors = [results[tissue][cut]['mean_rel_error']
                           for cut in results[tissue]]
            f.write(f'### {tissue.upper()} Predictions:\n')
            f.write(f'- Mean Error Across Cuts: {np.mean(tissue_errors):.2f}%\n')
            f.write(f'- Best Performing Cut: {min(results[tissue].items(), key=lambda x: x[1]["mean_rel_error"])[0]}\n')
            f.write(f'- Most Challenging Cut: {max(results[tissue].items(), key=lambda x: x[1]["mean_rel_error"])[0]}\n\n')

        # Detailed results by tissue type and cut
        f.write('## Detailed Results by Cut\n\n')
        for tissue in results:
            f.write(f'### {tissue.upper()}\n\n')
            f.write('| Cut | Mean Error (%) | Median Error (%) | Max Error (%) | Std Error (%) |\n')
            f.write('|-----|---------------|-----------------|--------------|---------------|\n')

            for cut in results[tissue]:
                metrics = results[tissue][cut]
                f.write(f'| {cut} | {metrics["mean_rel_error"]:.2f} | '
                       f'{metrics["median_rel_error"]:.2f} | {metrics["max_rel_error"]:.2f} | '
                       f'{metrics["std_rel_error"]:.2f} |\n')
            f.write('\n')

def main():
    print("Loading trained models...")
    predictor = BeefVolumePredictor.load_models()

    print("Loading evaluation data...")
    data = pd.read_csv('data/ct_composition.csv')

    print("Evaluating model performance...")
    results, predictions_df = evaluate_predictions(predictor, data)

    print("Generating visualizations...")
    plot_error_distributions(predictions_df, 'outputs')
    plot_prediction_scatter(predictions_df, 'outputs')

    print("Generating summary report...")
    generate_summary_report(results, predictions_df, 'outputs')

    print("\nAnalysis complete! Check the 'outputs' directory for the full report and visualizations.")

if __name__ == "__main__":
    main()
