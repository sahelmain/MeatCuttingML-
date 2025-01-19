import os
import torch
from typing import Dict, Any

from src.data.dataset import CoreSegmentationDataset
from src.models.segformer import CoreSegformerModel
from src.training.train import train_model
from src.inference.predict import predict

def main():
    """Main entry point for training and inference."""
    # Training configuration
    config: Dict[str, Any] = {
        # Data paths
        'train_image_dir': 'data/train/images',
        'train_mask_dir': 'data/train/masks',
        'val_image_dir': 'data/val/images',
        'val_mask_dir': 'data/val/masks',
        'test_image_dir': 'data/test/images',
        'save_dir': 'outputs',
        
        # Model parameters
        'num_classes': 3,
        'encoder_name': 'nvidia/mit-b2',
        'pretrained': True,
        
        # Training parameters
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'num_workers': 4,
        'max_grad_norm': 1.0,
        'gradient_accumulation_steps': 1,
        
        # Scheduler parameters
        'scheduler_T0': 10,
        'scheduler_T_mult': 2,
        'min_lr': 1e-6,
        
        # Early stopping
        'patience': 10,
        
        # Wandb logging
        'use_wandb': True,
        'wandb_project': 'core_segmentation',
        'wandb_run_name': 'segformer_b2',
        
        # Inference parameters
        'checkpoint_path': 'outputs/best_model.pth',
        'use_tta': True,
        'save_probabilities': True,
        'output_dir': 'outputs/predictions'
    }
    
    # Create output directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Training
    train_model(config)
    
    # Inference
    predict(config)

if __name__ == '__main__':
    main() 