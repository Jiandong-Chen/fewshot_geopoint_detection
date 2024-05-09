from ultralytics import YOLO
import os


model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
results = model.train(data='/home/jusun/chen8111/ondemand/Darpa/point_symbols.yaml', epochs=100, imgsz=1024, device='0')



# # Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model
# # Resume training
# results = model.train(resume=True)




