import os
import json
from PIL import Image
import PIL.Image
import glob

PIL.Image.MAX_IMAGE_PIXELS = None

def crop_and_save_image(image_path, coordinates, output_path):
    # Open the image
    original_image = Image.open(image_path)

    # Extract coordinates
    (x1, y1), (x2, y2) = coordinates

    upper_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    # Crop the image
    cropped_image = original_image.crop((*upper_left, *bottom_right))

    # Save the cropped image as JPEG
    cropped_image.save(output_path, "JPEG")

def crop_legned(folder_path, output_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            map_ = filename.split(".")[0]
            mapname = map_ + ".tif"
            map_path = os.path.join(folder_path, mapname)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # print(data.keys())
                for shape_dict in data["shapes"]:
                    if shape_dict["label"].endswith("_pt"):
                        coordinates = shape_dict["points"]
                        img_name = map_ + "_label_" + shape_dict["label"] + ".jpeg"
                        output = os.path.join(output_path, img_name)
                        crop_and_save_image(map_path, coordinates, output)



