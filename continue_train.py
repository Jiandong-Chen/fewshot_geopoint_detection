from ultralytics import YOLO
import os


model = YOLO('/home/jusun/chen8111/ondemand/Darpa/runs/detect/train/weights/last.pt')
results = model.train(resume = True)



# # Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model
# # Resume training
# results = model.train(resume=True)




