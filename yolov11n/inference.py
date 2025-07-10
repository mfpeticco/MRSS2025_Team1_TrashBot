#!/usr/bin/env python3
"""
Underwater Trash Detection - Real-time Camera Inference
Based on YOLOv8 model for robot integration
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import os
import time
from pathlib import Path

class TrashDetector:
    def __init__(self, model_path, confidence_threshold=0.5, device='auto'):
        """
        Initialize the trash detector
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load the model
        self.model = self._load_model()
        
        # Class names (based on your dataset)
        self.class_names = {
            0: 'trash',
            1: 'plastic',
            2: 'debris'
        }
        
        # Colors for bounding boxes
        self.colors = {
            0: (0, 0, 255),    # Red for trash
            1: (0, 255, 0),    # Green for plastic
            2: (255, 0, 0)     # Blue for debris
        }
        
    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Using device: {model.device}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_trash(self, frame):
        """
        Detect trash in a single frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            tuple: (annotated_frame, detections)
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Process results
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Store detection info
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, 'unknown')
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    color = self.colors.get(class_id, (255, 255, 255))
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{self.class_names.get(class_id, 'unknown')}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame, detections
    
    def run_camera_inference(self, camera_id=0, save_output=False, output_dir="output"):
        """
        Run real-time inference on camera stream
        
        Args:
            camera_id (int): Camera device ID (0 for default camera)
            save_output (bool): Whether to save output frames
            output_dir (str): Directory to save output frames
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create output directory if saving
        if save_output:
            Path(output_dir).mkdir(exist_ok=True)
            frame_count = 0
        
        print("Starting camera inference...")
        print("Press 'q' to quit, 's' to save current frame")
        
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Run detection
                annotated_frame, detections = self.detect_trash(frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update every 30 frames
                    elapsed_time = time.time() - start_time
                    fps = fps_counter / elapsed_time
                    print(f"FPS: {fps:.2f}")
                
                # Display detection count
                detection_text = f"Detections: {len(detections)}"
                cv2.putText(annotated_frame, detection_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Underwater Trash Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_output:
                    # Save current frame
                    filename = f"{output_dir}/frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Frame saved: {filename}")
                    frame_count += 1
                
                # Print detections (for robot integration)
                if detections:
                    print(f"Found {len(detections)} objects:")
                    for i, det in enumerate(detections):
                        print(f"  {i+1}. {det['class_name']} - Confidence: {det['confidence']:.2f} - "
                              f"BBox: {det['bbox']}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Underwater Trash Detection - Camera Inference')
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model (.pt file)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save', action='store_true', help='Save output frames')
    parser.add_argument('--output-dir', default='output', help='Output directory for saved frames')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize detector
    detector = TrashDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Run camera inference
    detector.run_camera_inference(
        camera_id=args.camera,
        save_output=args.save,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()