import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create output directory if it doesn't exist
import os
os.makedirs('outputs', exist_ok=True)

# Read the data
df = pd.read_csv('data/CT_wts_vols (1).csv')

def analyze_tissue_relationships(df, tissue_type):
    """Analyze relationships between weights and volumes for a specific tissue type"""
    # Get weight and volume columns for the tissue type
    weight_cols = [col for col in df.columns if col.startswith(f'tot_{tissue_type}_') and not col.endswith('vol')]
    volume_cols = [col for col in df.columns if col.startswith(f'tot_{tissue_type}_vol_')]

    # Create correlation matrix
    tissue_data = df[weight_cols + volume_cols]
    corr = tissue_data.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Matrix for {tissue_type.capitalize()} Measurements')
    plt.tight_layout()
    plt.savefig(f'outputs/{tissue_type}_correlations.png')
    plt.close()

    # Calculate weight-to-volume ratios
    cuts = ['chuckbrisket', 'chuckclod', 'chuckroll', 'loinwing', 'plate', 'rib', 'round', 'sirloin']
    ratios = {}
    for cut in cuts:
        weight_col = f'tot_{tissue_type}_{cut}'
        vol_col = f'tot_{tissue_type}_vol_{cut}'
        ratio = df[weight_col] / df[vol_col]
        ratios[cut] = ratio.mean()

    return ratios

def plot_tissue_distributions(df, tissue_type):
    """Plot distributions of weights and volumes for each cut"""
    cuts = ['chuckbrisket', 'chuckclod', 'chuckroll', 'loinwing', 'plate', 'rib', 'round', 'sirloin']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot weight distributions
    weight_data = pd.DataFrame()
    for cut in cuts:
        weight_data[cut] = df[f'tot_{tissue_type}_{cut}']
    weight_data.boxplot(ax=ax1)
    ax1.set_title(f'{tissue_type.capitalize()} Weights by Cut')
    ax1.set_ylabel('Weight')
    ax1.tick_params(axis='x', rotation=45)

    # Plot volume distributions
    volume_data = pd.DataFrame()
    for cut in cuts:
        volume_data[cut] = df[f'tot_{tissue_type}_vol_{cut}']
    volume_data.boxplot(ax=ax2)
    ax2.set_title(f'{tissue_type.capitalize()} Volumes by Cut')
    ax2.set_ylabel('Volume')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'outputs/{tissue_type}_distributions.png')
    plt.close()

def analyze_composition_relationships():
    """Analyze relationships between different tissue types"""
    # Calculate total weights for each tissue type
    tissue_types = ['bone', 'fat', 'musc']
    for tissue in tissue_types:
        weight_cols = [col for col in df.columns if col.startswith(f'tot_{tissue}_') and not col.endswith('vol')]
        df[f'total_{tissue}_weight'] = df[weight_cols].sum(axis=1)

    # Create correlation matrix for total tissue weights
    total_weights = df[[f'total_{tissue}_weight' for tissue in tissue_types]]
    corr = total_weights.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Tissue Types')
    plt.tight_layout()
    plt.savefig('outputs/tissue_type_correlations.png')
    plt.close()

def main():
    print("Analyzing tissue relationships...")

    # Analyze each tissue type
    tissue_types = ['bone', 'fat', 'musc']
    density_ratios = {}

    for tissue in tissue_types:
        print(f"\nAnalyzing {tissue} measurements:")
        ratios = analyze_tissue_relationships(df, tissue)
        density_ratios[tissue] = ratios
        plot_tissue_distributions(df, tissue)

        print(f"{tissue.capitalize()} density ratios (weight/volume) for each cut:")
        for cut, ratio in ratios.items():
            print(f"  {cut}: {ratio:.4f}")

    # Analyze relationships between tissue types
    analyze_composition_relationships()

    print("\nAnalysis complete! Check the 'outputs' directory for visualizations.")

    # Print summary statistics
    print("\nSummary Statistics:")
    for tissue in tissue_types:
        weight_cols = [col for col in df.columns if col.startswith(f'tot_{tissue}_') and not col.endswith('vol')]
        total_weight = df[weight_cols].sum(axis=1)
        print(f"\n{tissue.capitalize()} Statistics:")
        print(f"  Total Weight: Mean = {total_weight.mean():.2f}, Std = {total_weight.std():.2f}")
        print(f"  Percentage of Total Mass: {(total_weight.mean() / total_weight.sum() * 100):.2f}%")

if __name__ == "__main__":
    main()
