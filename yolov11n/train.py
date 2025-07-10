import os
import torch
from ultralytics import YOLO

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
    
    # Initialize model
    model = YOLO('yolo11n.pt')  # Load pretrained model
    
    # Set device to MPS if available
    device = 'mps' if mps_available else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train the model
    results = model.train(
        data='./data/utd2-7/data.yaml',  # Path to dataset YAML
        epochs=200,
        imgsz=416,
        batch=512,
        lr0=0.0025,
        optimizer='Adam',
        plots=True,
        save=True,
        project='./models',
        name='yolov11n_trash_detection',
        device=device,
        autoanchor=True
    )
    
    print("Training completed!")
    print(f"Model saved to: ./models/yolov11n_trash_detection")
    print(f"Training device used: {device}")

if __name__ == "__main__":
    main()