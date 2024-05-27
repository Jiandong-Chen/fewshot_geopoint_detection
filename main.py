from ultralytics import YOLO
import os
from PIL import Image
from generate_sythentic_data import generate_sythentic_data
import random
import shutil
import yaml
from map2patch import crop_map_main

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# os.chdir("/Users/dong_dong_dong/Downloads/Darpa/fewshot")
# current_directory = os.getcwd()
# print("Current working directory:", current_directory)

# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# results = model.train(data='/Users/dong_dong_dong/Downloads/Darpa/fewshot/point_symbols.yaml', epochs=100, imgsz=1024, device='mps')

# Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model
# # Resume training
# results = model.train(resume=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse all the required paths.")

    parser.add_argument('--rawmap_path', type=str, default="/Users/dong_dong_dong/Downloads/Darpa/eval/eval_data_perfomer",
                        help='Path of a list of prediction maps')
    parser.add_argument('--main_path', type=str, default="/Users/dong_dong_dong/Downloads/Darpa/fewshot",
                        help='Main path for the project')
    parser.add_argument('--generated_meta_info', type=str, default='/Users/dong_dong_dong/Downloads/Darpa/legend_item_description_outputs/evaluation',
                        help='Path to the generated metainfo contains map area coordinates')
    parser.add_argument('--sythentic_data_output_path', type=str, default='/Users/dong_dong_dong/Downloads/Darpa/fewshot/sythentic_pred',
                        help='Path to the generated synthetic data')
    parser.add_argument('--map_patch_output_dir', type=str, default='/Users/dong_dong_dong/Downloads/Darpa/cropped_maps',
                        help='Directory for cropped map patch outputs')
    parser.add_argument('--predict_output_dir', type=str, default='/Users/dong_dong_dong/Downloads/Darpa/cropped_maps',
                        help='Directory for prediction outputs')
    parser.add_argument('--final_output_dir_root', type=str, default='/Users/dong_dong_dong/Downloads/Darpa/fewshot',
                        help='Root directory for final outputs')
    parser.add_argument('--baseline_model_path', type=str, default='/Users/dong_dong_dong/Downloads/Darpa/fewshot/runs/detect/train/weights/best.pt',
                        help='Path to the baseline model weights')

    args = parser.parse_args()

    return args

args = parse_arguments()

folder_path = args.rawmap_path
main_path = args.main_path
generated_meta_info = args.generated_meta_info
sythentic_data_path = args.sythentic_data_output_path
map_patch_output_dir = args.map_patch_output_dir
predict_output_dir = args.predict_output_dir
final_output_dir_root = args.final_output_dir_root
baseline_model_path = args.baseline_model_path

legend_path = os.path.join(main_path, 'Cropped_legend')
cropped_path = os.path.join(legend_path, 'ValLabels')
processed_legend_path = os.path.join(legend_path, 'ValLabels_processed')
sqaure_legend_path = os.path.join(legend_path, 'square_labels')

os.makedirs(cropped_path, exist_ok=True)
os.makedirs(processed_legend_path, exist_ok=True)
os.makedirs(sqaure_legend_path, exist_ok=True)

def create_yaml_format_folder(root):
    # Remove all contents of synthetic_pred directory, if it exists
    if os.path.exists(root):
        shutil.rmtree(root)

    # Recreate the synthetic_pred directory
    os.makedirs(root, exist_ok=True)

    # Define the images and labels data paths
    synthetic_images_data = os.path.join(root, 'images')
    synthetic_labels_data = os.path.join(root, 'labels')

    # Create the images and labels data directories
    os.makedirs(synthetic_images_data, exist_ok=True)
    os.makedirs(synthetic_labels_data, exist_ok=True)

    # Define the train, test, and val directories for images
    synthetic_image_train_data = os.path.join(synthetic_images_data, 'train')
    synthetic_image_test_data = os.path.join(synthetic_images_data, 'test')
    synthetic_image_val_data = os.path.join(synthetic_images_data, 'val')

    # Define the train, test, and val directories for labels
    synthetic_labels_train_data = os.path.join(synthetic_labels_data, 'train')
    synthetic_labels_test_data = os.path.join(synthetic_labels_data, 'test')
    synthetic_labels_val_data = os.path.join(synthetic_labels_data, 'val')

    # Create the train, test, and val directories for images and labels
    os.makedirs(synthetic_image_train_data, exist_ok=True)
    os.makedirs(synthetic_image_test_data, exist_ok=True)
    os.makedirs(synthetic_image_val_data, exist_ok=True)
    os.makedirs(synthetic_labels_train_data, exist_ok=True)
    os.makedirs(synthetic_labels_test_data, exist_ok=True)
    os.makedirs(synthetic_labels_val_data, exist_ok=True)

    print(f"Directories created in {root}: images (train, test, val) and labels (train, test, val)")

# crop_legned(folder_path, cropped_path)
# target_color = [255, 255, 255]
# clean_background_and_remove_text(cropped_path, processed_path, target_color, threshold=50, crop=False, use_gpu=False)



# crop prediction maps to get the patches
# for img_name in os.listdir(folder_path):
#     if folder_path != ".DS_Store" and img_name.endswith('.tif'):
#         mapname = img_name.split('.')[0]
#         map_path = os.path.join(folder_path, img_name)
#         map_meta = os.path.join(generated_meta_info, mapname + '_*.json')
#         crop_map_main(map_path, map_meta, 1024, 1024, map_patch_output_dir)

def move_images(source_folder, destination_folder):
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over all files in the source folder
    for file_name in os.listdir(source_folder):
        # Get the full path of the file
        file_path = os.path.join(source_folder, file_name)

        # Check if the file is an image (you can add more file types as needed)
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            # Construct the destination path
            dest_path = os.path.join(destination_folder, file_name)

            # Move the file
            shutil.move(file_path, dest_path)

def resize_and_center_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Get original dimensions
    width, height = img.size

    # Calculate the dimensions for the square
    max_dimension = max(width, height)
    new_width = new_height = max_dimension

    # Create a new blank square image with white background
    square_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Calculate the position to paste the original image
    paste_x = (new_width - width) // 2
    paste_y = (new_height - height) // 2

    # Paste the original image onto the center of the square image
    square_img.paste(img, (paste_x, paste_y))

    # Resize the square image to the desired output size
    square_img = square_img.resize((max_dimension, max_dimension))

    # Save or display the resulting image
    output_folder = os.path.dirname(image_path)
    # output_path = os.path.join(output_folder, output_filename)
    output_path = os.path.join(output_folder, "..", "square_labels")
    output_path_legend = os.path.join(output_path, os.path.basename(image_path).split("_label_")[0], os.path.basename(image_path).split(".")[0])
    os.makedirs(output_path_legend, exist_ok=True)
    output_filename = os.path.join(output_path_legend, os.path.basename(image_path).split(".")[0] + ".png")
    square_img.save(output_filename, "PNG")

# for img in os.listdir(processed_legend_path):
#     if img != ".DS_Store":
#         pt_img_path = os.path.join(processed_legend_path, img)
#         resize_and_center_image(pt_img_path)
index = 0
index_dict = {}
for square_map in os.listdir(sqaure_legend_path):
    if square_map != ".DS_Store":
        square_map_legend = os.path.join(sqaure_legend_path, square_map)
        print(square_map_legend)
        index_name_dict = {}
        index = 0
        for square_legend in os.listdir(square_map_legend):
            if square_legend != ".DS_Store":
                cur_legend = os.path.join(square_map_legend, square_legend)
                mapname = square_legend.split("_label_")[0]
                pt_name = square_legend
                SYMBOL_BASE_SIZE = [(10, 10),(20,20),(30,30),(40,40),(50,50),(60, 60)]
                index_dict[index] = pt_name
                max_num_synthetic_images = 1
                print(cur_legend)
                generate_sythentic_data(cur_legend, index, SYMBOL_BASE_SIZE, sythentic_data_path, max_num_synthetic_images)

                index_name_dict[index] = pt_name
                cur_legend_info_path = os.path.join(sythentic_data_path, 'point_synthetic_maps',str(index))
                cur_legend_image_path = os.path.join(cur_legend_info_path, 'images')
                cur_legend_label_path = os.path.join(cur_legend_info_path, 'labels')

                # store all of the sythentic legend info for a map
                all_legend_info_path = os.path.join(sythentic_data_path, 'point_synthetic_maps', mapname)
                all_legend_image_path = os.path.join(all_legend_info_path, 'images')
                all_legend_label_path = os.path.join(all_legend_info_path, 'labels')
                # Define the train, test, and val directories for images
                synthetic_image_train_data = os.path.join(all_legend_image_path, 'train')
                synthetic_image_test_data = os.path.join(all_legend_image_path, 'test')
                synthetic_image_val_data = os.path.join(all_legend_image_path, 'val')

                # Define the train, test, and val directories for labels
                synthetic_labels_train_data = os.path.join(all_legend_label_path, 'train')
                synthetic_labels_test_data = os.path.join(all_legend_label_path, 'test')
                synthetic_labels_val_data = os.path.join(all_legend_label_path, 'val')

                os.makedirs(synthetic_image_train_data, exist_ok=True)
                os.makedirs(synthetic_image_test_data, exist_ok=True)
                os.makedirs(synthetic_image_val_data, exist_ok=True)
                os.makedirs(synthetic_labels_train_data, exist_ok=True)
                os.makedirs(synthetic_labels_test_data, exist_ok=True)
                os.makedirs(synthetic_labels_val_data, exist_ok=True)

                images = [img for img in os.listdir(cur_legend_image_path) if img.endswith(('jpg', 'jpeg', 'png'))]
                random.shuffle(images)
                split_index = int(0.2 * len(images))
                val_images = images[:split_index]
                train_images = images[split_index:]

                # Move validation images to val folder
                for img in val_images:
                    source_img_path = os.path.join(cur_legend_info_path, 'images',img)
                    source_label_path = os.path.join(cur_legend_info_path, 'labels',img.split('.')[0] + ".txt")
                    val_img_path = os.path.join(all_legend_info_path,'images','val',str(index)+img)
                    val_label_path = os.path.join(all_legend_info_path, 'labels', 'val', str(index)+img.split('.')[0] + ".txt")

                    shutil.move(source_img_path, val_img_path)
                    shutil.move(source_label_path, val_label_path)

                # Move training images to train folder
                for img in train_images:
                    source_img_path = os.path.join(cur_legend_info_path, 'images',img)
                    source_label_path = os.path.join(cur_legend_info_path, 'labels',img.split('.')[0] + ".txt")
                    train_img_path = os.path.join(all_legend_info_path,'images','train',str(index)+img)
                    train_label_path = os.path.join(all_legend_info_path, 'labels', 'train', str(index)+img.split('.')[0] + ".txt")

                    shutil.move(source_img_path, train_img_path)
                    shutil.move(source_label_path, train_label_path)

                index += 1

        print("Images have been split into training and validation sets.")

        test_source_folder = os.path.join(map_patch_output_dir, mapname + "_g1024_s1024")
        test_destination_folder = os.path.join(all_legend_info_path,'images','test')

        move_images(test_source_folder, test_destination_folder)

        yaml_data = {
            'path': all_legend_info_path,
            'train': os.path.join(all_legend_info_path,'images','train'),
            'val': os.path.join(all_legend_info_path,'images','val'),
            'test': test_destination_folder,  # Test images (optional), you can add a path if needed
            'names': index_name_dict
        }

        # Define the path for the YAML file
        yaml_file_path = os.path.join(all_legend_info_path, "fewshot_dataset_info.yaml")

        # Save the dictionary data to the YAML file
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        print(f"YAML file has been created at {yaml_file_path}.")

        model = YOLO(baseline_model_path)  # load a pretrained model (recommended for training)
        results = model.train(data=yaml_file_path, epochs=20, imgsz=1024, device='0')

        # index += 1
        fewshot_prediction(model_weights_dir, folder_path, map_patch_output_dir, predict_output_dir, final_output_dir_root)

