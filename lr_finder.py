import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torch_lr_finder import LRFinder
import numpy as np

def main():
    # Setup
    print(f"Working directory: {os.getcwd()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for MPS (Metal Performance Shaders) availability
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS (Apple Metal) available: {mps_available}")
    
    # Also check for CUDA as fallback
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs("./lr_finder_results", exist_ok=True)
    
    # Initialize model
    model = YOLO('yolo11n.pt')  # Load pretrained model
    
    # Set device to MPS if available
    device = 'mps' if mps_available else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a temporary training configuration to get the model's PyTorch components
    print("Setting up model for learning rate finding...")
    
    # We need to create a simple training setup to extract the model and optimizer
    # YOLO handles this internally, so we'll use a different approach
    try:
        # Create a custom training loop for LR finding
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.cfg import get_cfg
        
        # Get default config and override with our settings
        cfg = get_cfg()
        cfg.update({
            'data': './data/utd2-7/data.yaml',
            'epochs': 1,  # We only need one epoch for LR finding
            'imgsz': 416,
            'batch': 512,
            'device': device,
            'optimizer': 'Adam',
            'lr0': 0.001,  # Starting LR for the finder
        })
        
        # Create trainer
        trainer = DetectionTrainer(cfg=cfg, overrides={})
        trainer.setup_model()
        trainer.set_model_attributes()
        
        # Get the model and optimizer
        pytorch_model = trainer.model
        optimizer = trainer.optimizer
        criterion = trainer.criterion
        
        # Create data loaders
        trainer.get_dataloader(trainer.trainset, batch_size=cfg.batch, rank=-1, mode='train')
        train_loader = trainer.train_loader
        
        print("Starting learning rate finding...")
        
        # Initialize LR Finder
        lr_finder = LRFinder(pytorch_model, optimizer, criterion, device=device)
        
        # Perform LR range test
        lr_finder.range_test(
            train_loader, 
            end_lr=1, 
            num_iter=100,
            step_mode='exp',
            smooth_f=0.05,
            diverge_th=5
        )
        
        # Plot and save results
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_finder.plot(ax=ax)
        plt.title('Learning Rate Finder Results')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = './lr_finder_results/lr_finder_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"LR Finder plot saved to: {plot_path}")
        
        # Get suggested learning rate
        suggested_lr = lr_finder.suggest_lr()
        print(f"\nSuggested learning rate: {suggested_lr}")
        
        # Save results to file
        results_path = './lr_finder_results/lr_finder_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"Learning Rate Finder Results\n")
            f.write(f"============================\n\n")
            f.write(f"Suggested Learning Rate: {suggested_lr}\n")
            f.write(f"Device used: {device}\n")
            f.write(f"Batch size: {cfg.batch}\n")
            f.write(f"Image size: {cfg.imgsz}\n")
            f.write(f"Dataset: {cfg.data}\n")
            f.write(f"Optimizer: {cfg.optimizer}\n")
        
        print(f"Results saved to: {results_path}")
        
        # Reset the model
        lr_finder.reset()
        
        # Show additional analysis
        print("\nLearning Rate Analysis:")
        print(f"- Suggested LR: {suggested_lr}")
        print(f"- You can try learning rates around: {suggested_lr * 0.1:.6f} to {suggested_lr * 10:.6f}")
        print(f"- Current train.py uses: 0.0025")
        
        if suggested_lr:
            if suggested_lr > 0.0025:
                print(f"- Consider increasing your learning rate to ~{suggested_lr:.6f}")
            else:
                print(f"- Consider decreasing your learning rate to ~{suggested_lr:.6f}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error during LR finding: {e}")
        print("You may need to install torch-lr-finder: pip install torch-lr-finder")
        
        # Alternative simple approach using YOLO's built-in functionality
        print("\nFalling back to YOLO's built-in LR optimization...")
        
        # Run a short training with lr finder
        results = model.train(
            data='./data/utd2-7/data.yaml',
            epochs=10,  # Short run for LR testing
            imgsz=416,
            batch=512,
            lr0=0.01,  # Start higher
            lrf=0.0001,  # End lower
            optimizer='Adam',
            plots=True,
            save=False,
            project='./lr_finder_results',
            name='lr_test',
            device=device
        )
        
        print("Check the training plots in ./lr_finder_results/lr_test/")
        print("Look for the learning rate vs loss plot to find optimal LR")

if __name__ == "__main__":
    main()
