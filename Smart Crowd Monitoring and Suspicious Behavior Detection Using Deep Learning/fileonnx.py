import torch
from ultralytics import YOLO

# Step 1: Load the YOLO model
model = YOLO("models/best.pt")  # Replace 'yolov8n.pt' with your YOLO model

# Step 2: Export the model to ONNX format
model.export(format='onnx')

print("Model exported to ONNX format successfully!")


#pip install ultralytics onnx onnxruntime
