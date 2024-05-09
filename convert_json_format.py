from ultralytics import YOLO
import os
from PIL import Image

def convert_coordinates(image_path, annotation_path, output_path):
    output_file_path = os.path.join(output_path, f"{os.path.basename(image_path)[:-4]}.txt")
    with open(annotation_path, 'r') as coordinates_file, open(output_file_path, 'w') as output_file:
        # Read each line from the file
        for line in coordinates_file:
            # Split the line into label and coordinates
            label, x1, y1, x2, y2 = map(int, line.strip().split(','))

            # Calculate center coordinates and dimensions
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2

            # Normalize the coordinates
            img = Image.open(image_path)
            img_width, img_height = img.size
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            output_file.write(f"{label} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")


# Example usage
cropped_patches = '/Users/dong_dong_dong/Downloads/Darpa/train/train_bbox/images'
annotation = '/Users/dong_dong_dong/Downloads/Darpa/train/train_bbox/annotation'
output_path = '/Users/dong_dong_dong/Downloads/Darpa/fewshot/datasets/ta1/labels/train'

for img_name in os.listdir(cropped_patches):
    if img_name.endswith('.jpg'):
        img_annotation = img_name[:-4] + ".txt"
        img_annotation_path = os.path.join(annotation, img_annotation)
        image_path = os.path.join(cropped_patches, img_name)
        convert_coordinates(image_path, img_annotation_path, output_path)


