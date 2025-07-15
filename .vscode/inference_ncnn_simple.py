import os
import random
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def get_random_image(valid_images_path):
    """Get a random image from the validation folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(valid_images_path).glob(f"*{ext}"))
        image_files.extend(Path(valid_images_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {valid_images_path}")
        return None
    
    return random.choice(image_files)

def process_image(model, image_path):
    """Process a single image and return results"""
    print(f"Selected image: {image_path.name}")
    
    # Run inference
    results = model(str(image_path))
    
    # Process results
    for r in results:
        # Plot results
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Save results
        # output_dir = Path("inference_results")
        # output_dir.mkdir(exist_ok=True)
        # output_path = output_dir / f"result_{image_path.name}"
        # cv2.imwrite(str(output_path), im_array)
        # print(f"Results saved to: {output_path}")
        
        # Print detection details
        if r.boxes is not None:
            print(f"Detected {len(r.boxes)} objects:")
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"  {i+1}. {class_name}: {confidence:.2f}")
        else:
            print("No objects detected")
        
        return im, image_path.name

def on_key(event):
    """Handle key press events"""
    if event.key == ' ':
        # Load new random image
        random_image = get_random_image("data/utd2-7/valid/images")
        if random_image:
            im, name = process_image(model, random_image)
            ax.clear()
            ax.imshow(im)
            ax.set_title(f"Trash Detection Results - {name}")
            ax.axis('off')
            plt.draw()
    elif event.key == 'q':
        plt.close()

def run_random_inference():
    global model, ax
    
    # Load the trained model once
    model_path = "models/yolov11n_trash_detection/weights/best_ncnn_model"
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    # Path to validation images
    valid_images_path = "data/utd2-7/valid/images"
    
    # Get first random image
    random_image = get_random_image(valid_images_path)
    if not random_image:
        return
    
    # Process first image
    im, name = process_image(model, random_image)
    
    # Create interactive plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(im)
    ax.set_title(f"Trash Detection Results - {name}")
    ax.axis('off')
    
    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add instructions
    fig.suptitle("Press SPACE for new random image, Q to quit", fontsize=14, y=0.02)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_random_inference()
