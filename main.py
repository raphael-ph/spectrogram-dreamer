import torch
import argparse
from pathlib import Path
from src.preprocessing.launch import launch
from src.dataset.spectrogram_dataset import SpectrogramDataset
from src.models import DreamerModel
from src.training import train
from src.utils.logger import get_logger

_logger = get_logger("main", level="INFO")

# run preprocessing pipeline
# launch()

def main():
    """Main entry point for training Dreamer model"""
    
    parser = argparse.ArgumentParser(description="Train Dreamer model on spectrograms")
    parser.add_argument("--spec-path", type=str, default="data/2_mel-spectrograms",
                        help="Path to spectrogram data")
    parser.add_argument("--style-path", type=str, default="data/3_style-vectors",
                        help="Path to style vectors")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--experiment-name", type=str, default="dreamer-spectrogram",
                        help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name")
    parser.add_argument("--h-state-size", type=int, default=200,
                        help="Size of deterministic state")
    parser.add_argument("--z-state-size", type=int, default=30,
                        help="Size of stochastic state")
    parser.add_argument("--action-size", type=int, default=128,
                        help="Size of style action vector")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with dummy data")
    
    args = parser.parse_args()
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _logger.info(f"Using device: {device}")
    _logger.info(f"MLflow tracking: mlruns/{args.experiment_name}")
    
    # Initialize model
    _logger.info("Initializing Dreamer model...")
    model = DreamerModel(
        h_state_size=args.h_state_size,
        z_state_size=args.z_state_size,
        action_size=args.action_size,
        embedding_size=256,
        aux_size=5,  # pitch, energy, delta-energy, spectral centroid, onset strength
        in_channels=1,
        cnn_depth=32
    )
    
    # Load dataset
    _logger.info("Loading dataset...")
    try:
        if not args.test_mode:
            dataset = SpectrogramDataset(args.spec_path, args.style_path)
            _logger.info(f"Dataset loaded with {len(dataset)} samples")
            
            # Show sample
            sample = dataset[0]
            _logger.info(f"Sample observation shape: {sample['observation'].shape}")
            _logger.info(f"Sample action shape: {sample['action'].shape}")
            _logger.info(f"Sample rewards shape: {sample['rewards'].shape}")
            
            # Start training with MLflow
            _logger.info("Starting training with MLflow tracking...")
            train(
                model, 
                dataset, 
                num_epochs=args.epochs, 
                batch_size=args.batch_size, 
                device=device,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                checkpoint_freq=args.checkpoint_freq
            )
            
            _logger.info("Training complete! Check mlruns/ directory for results.")
            _logger.info("To view results: mlflow ui")
        
    except FileNotFoundError as e:
        _logger.error(f"Dataset not found: {e}")
        _logger.info("Please run the preprocessing pipeline first or adjust paths")
        _logger.info("You can test individual components with --test-mode")
        args.test_mode = True
    
    if args.test_mode:
        # Test mode - create dummy data
        _logger.info("Running in test mode with dummy data...")
        batch_size = 4
        seq_len = 50
        dummy_obs = torch.randn(batch_size, seq_len, 1, 64, 10)
        dummy_actions = torch.randn(batch_size, seq_len, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_obs, dummy_actions, compute_loss=False)
            _logger.info(f"Reconstructed shape: {output['reconstructed'].shape}")
            _logger.info(f"h_states shape: {output['h_states'].shape}")
            _logger.info(f"z_states shape: {output['z_states'].shape}")
            _logger.info("Model test successful!")


if __name__ == "__main__":
    main()