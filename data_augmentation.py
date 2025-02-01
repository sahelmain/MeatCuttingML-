import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAugmenter:
    """Handles data augmentation for beef carcass composition prediction."""

    def __init__(self, random_seed: int = 42):
        """Initialize with optional random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def augment_data(self, X: pd.DataFrame, y_dict: Dict,
                    augmentation_factor: int = 10,
                    noise_scale: float = 0.05) -> Tuple[pd.DataFrame, Dict]:
        """
        Augment the input dataset by generating synthetic samples.

        Parameters:
        ----------
        X : pd.DataFrame
            Original predictor variables (non-volume measurements)
        y_dict : Dict
            Original target variables (CT-derived volumes) organized by tissue and cut
        augmentation_factor : int
            How many total samples we want per original sample
        noise_scale : float
            Fraction of the feature's standard deviation to use as noise

        Returns:
        -------
        Tuple[pd.DataFrame, Dict]
            Augmented predictors DataFrame and targets dictionary
        """
        logger.info(f"Starting data augmentation: {X.shape[0]} samples → {X.shape[0] * augmentation_factor} samples")

        # List to hold augmented X (start with original)
        augmented_X_list = [X.copy()]

        # For each tissue/cut, prepare a list starting with original targets
        augmented_y_list = {}
        tissue_types = list(y_dict.keys())

        for tissue in tissue_types:
            augmented_y_list[tissue] = {}
            for cut in y_dict[tissue].keys():
                augmented_y_list[tissue][cut] = [y_dict[tissue][cut].copy()]

        # Compute per-feature standard deviation (for adding noise)
        feature_std = X.std().to_numpy()  # Convert to numpy array

        # Generate synthetic samples
        num_samples = X.shape[0]
        for i in range(augmentation_factor - 1):
            # Generate noise for features using numpy operations
            noise = np.random.randn(X.shape[0], X.shape[1]) * (feature_std * noise_scale)
            X_synth = X.values + noise  # Convert to numpy for addition
            X_synth = pd.DataFrame(X_synth, columns=X.columns)  # Convert back to DataFrame
            augmented_X_list.append(X_synth)

            # Generate noise for targets
            for tissue in tissue_types:
                for cut in y_dict[tissue].keys():
                    y_orig = y_dict[tissue][cut]
                    y_std = float(y_orig.std())  # Convert to float for multiplication
                    y_noise = np.random.randn(num_samples) * (y_std * noise_scale)
                    y_synth = y_orig.values + y_noise  # Convert to numpy for addition
                    augmented_y_list[tissue][cut].append(pd.Series(y_synth))

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1} synthetic copies...")

        # Concatenate all synthetic copies
        X_augmented = pd.concat(augmented_X_list, axis=0).reset_index(drop=True)

        y_augmented = {}
        for tissue in tissue_types:
            y_augmented[tissue] = {}
            for cut in y_dict[tissue].keys():
                y_augmented[tissue][cut] = pd.concat(
                    augmented_y_list[tissue][cut],
                    axis=0
                ).reset_index(drop=True)

        logger.info(f"Data augmentation complete: {X_augmented.shape[0]} total samples")
        return X_augmented, y_augmented

    def validate_augmented_data(self, X_orig: pd.DataFrame, y_dict_orig: Dict,
                              X_aug: pd.DataFrame, y_dict_aug: Dict) -> bool:
        """
        Validate that augmented data maintains reasonable statistical properties.

        Parameters:
        ----------
        X_orig, y_dict_orig : Original data
        X_aug, y_dict_aug : Augmented data

        Returns:
        -------
        bool : Whether validation passed
        """
        try:
            # Check shapes
            orig_samples = X_orig.shape[0]
            aug_samples = X_aug.shape[0]
            if aug_samples <= orig_samples:
                logger.error("Augmented data not larger than original")
                return False

            # Check feature means (should be similar)
            orig_means = X_orig.mean()
            aug_means = X_aug.mean()
            mean_diff = np.abs(orig_means - aug_means)
            if (mean_diff > 0.1 * orig_means).any():
                logger.warning("Large deviation in feature means detected")

            # Check feature correlations (should be preserved)
            orig_corr = X_orig.corr()
            aug_corr = X_aug.corr()
            corr_diff = np.abs(orig_corr - aug_corr)
            if (corr_diff > 0.2).any().any():
                logger.warning("Large deviation in feature correlations detected")

            # Check target distributions
            for tissue in y_dict_orig.keys():
                for cut in y_dict_orig[tissue].keys():
                    y_orig = y_dict_orig[tissue][cut]
                    y_aug = y_dict_aug[tissue][cut]

                    # Compare means
                    mean_diff_pct = abs(y_orig.mean() - y_aug.mean()) / y_orig.mean() * 100
                    if mean_diff_pct > 5:
                        logger.warning(f"Large mean difference ({mean_diff_pct:.1f}%) "
                                     f"in {tissue} volume for {cut}")

                    # Compare standard deviations
                    std_diff_pct = abs(y_orig.std() - y_aug.std()) / y_orig.std() * 100
                    if std_diff_pct > 10:
                        logger.warning(f"Large std difference ({std_diff_pct:.1f}%) "
                                     f"in {tissue} volume for {cut}")

            logger.info("Data augmentation validation complete")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False

def main():
    """Test the data augmentation functionality."""
    try:
        # Load original data
        data_path = "data/ct_composition.csv"
        df = pd.read_csv(data_path)

        # Separate features and targets (simplified example)
        X = df[[col for col in df.columns if not 'vol' in col.lower()
                and not 'unnamed' in col.lower()
                and col != 'id' and col != '']]

        # Create a simple y_dict for testing
        y_dict = {
            'bone': {'sirloin': df['tot_bone_vol_sirloin']},
            'fat': {'sirloin': df['tot_fat_vol_sirloin']},
            'musc': {'sirloin': df['tot_musc_vol_sirloin']}
        }

        # Create augmenter and test
        augmenter = DataAugmenter()
        X_aug, y_aug = augmenter.augment_data(X, y_dict)

        # Validate results
        is_valid = augmenter.validate_augmented_data(X, y_dict, X_aug, y_aug)

        if is_valid:
            logger.info("Test completed successfully")
        else:
            logger.error("Test failed validation")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
