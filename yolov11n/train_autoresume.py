import os
import torch
from ultralytics import YOLO
import glob
import time

'''
This script trains a YOLOv11n model for trash detection with automatic resuming from the latest checkpoint.
This is mainly done because of weirdness with training on apple silicon with MPS.
'''

def find_latest_checkpoint(project_path, run_name):
    """Find the latest checkpoint file in the training directory"""
    checkpoint_pattern = os.path.join(project_path, run_name, "weights", "last.pt")
    if os.path.exists(checkpoint_pattern):
        return checkpoint_pattern
    return None

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
    os.makedirs("./models", exist_ok=True)
    
    # Training parameters
    project_path = './models'
    run_name = 'yolov11n_trash_detection'
    max_retries = 3
    
    # Set device to MPS if available
    device = 'mps' if mps_available else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check for existing checkpoint
    checkpoint_path = find_latest_checkpoint(project_path, run_name)
    
    if checkpoint_path:
        print(f"Found existing checkpoint: {checkpoint_path}")
        print("Resuming training from checkpoint...")
        model = YOLO(checkpoint_path)
    else:
        print("No checkpoint found, starting fresh training...")
        model = YOLO('yolo11n.pt')  # Load pretrained model
    
    # Training loop with retry logic
    for attempt in range(max_retries):
        try:
            print(f"Training attempt {attempt + 1}/{max_retries}")
            
            # Train the model
            if checkpoint_path:
                # If resuming, only pass the essential arguments. model was updated to be the checkpoint
                results = model.train(
                    data='./data/utd2-7/data.yaml',
                    resume=True
                )
            else:
                # Full parameter set for fresh training
                results = model.train(
                    data='./data/utd2-7/data.yaml',  # Path to dataset YAML
                    epochs=200,
                    imgsz=416,
                    batch=512,
                    lr0=0.0025,
                    optimizer='Adam',
                    plots=True,
                    save=True,
                    project=project_path,
                    name=run_name,
                    device=device
                )
            
            print("Training completed successfully!")
            print(f"Model saved to: {project_path}/{run_name}")
            print(f"Training device used: {device}")
            break
            
        except Exception as e:
            print(f"Training failed on attempt {attempt + 1}: {str(e)}")
            
            if attempt < max_retries - 1:
                print("Attempting to resume from checkpoint...")
                # Look for checkpoint again in case it was created during the failed run
                checkpoint_path = find_latest_checkpoint(project_path, run_name)
                if checkpoint_path:
                    print(f"Loading checkpoint: {checkpoint_path}")
                    model = YOLO(checkpoint_path)
                else:
                    print("No checkpoint found, will retry with original model")
                    model = YOLO('yolo11n.pt')
                
                # Wait a bit before retrying
                time.sleep(5)
            else:
                print("Maximum retry attempts reached. Training failed.")
                raise e

if __name__ == "__main__":
    main()