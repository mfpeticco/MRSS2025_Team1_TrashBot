# export_ncnn.py
from ultralytics import YOLO
import os
import shutil
import subprocess

# 1. load your trained model
model = YOLO('models/yolov11n_trash_detection/weights/best.pt')

# Define destination directory
dest_dir = 'ncnn_test'
os.makedirs(dest_dir, exist_ok=True)

print("Exporting to ONNX...")
# Export to ONNX with specific settings for NCNN
model.export(
    format='onnx',
    imgsz=416,
    batch=1,
    simplify=True,
    opset=11,  # Use opset 11 for better NCNN compatibility
    dynamic=False  # Ensure static shapes
)

print("Exporting to NCNN...")
# Export to NCNN
model.export(
    format='ncnn',
    imgsz=416,
    batch=1,
    simplify=True
)

# Move exported files to destination directory
source_dir = 'models/yolov11n_trash_detection/weights'
for file in os.listdir(source_dir):
    if file.endswith('.onnx') or file.endswith('.param') or file.endswith('.bin') or file.endswith('.torchscript'):
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(source_path, dest_path)
        print(f"Moved {file} to {dest_dir}")
    elif file == 'best_ncnn_model':
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(source_path, dest_path)
        print(f"Moved {file} directory to {dest_dir}")

# Try to optimize the NCNN model if ncnnoptimize is available
param_file = os.path.join(dest_dir, 'best.param')
bin_file = os.path.join(dest_dir, 'best.bin')
opt_param = os.path.join(dest_dir, 'best-opt.param')
opt_bin = os.path.join(dest_dir, 'best-opt.bin')

if os.path.exists(param_file) and os.path.exists(bin_file):
    try:
        print("Optimizing NCNN model...")
        result = subprocess.run([
            'ncnnoptimize', 
            param_file, 
            bin_file, 
            opt_param, 
            opt_bin, 
            '65536'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("NCNN model optimization successful!")
            print(f"Optimized files: {opt_param}, {opt_bin}")
        else:
            print(f"NCNN optimization failed: {result.stderr}")
    except FileNotFoundError:
        print("ncnnoptimize not found. Using unoptimized model.")
        print("Install ncnn tools for optimization.")

print("Export complete!")
