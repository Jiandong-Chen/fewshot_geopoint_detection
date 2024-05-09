from generate_Labels import *
from Seperate_symbol_text import *
import os
import sys
import shutil
from data.data_loader import *
from data.data_utils import build_map_mask, build_point_mask, build_text_mask
from func import sample_symbol_image, create_symbol_images, synthesize
import time
Image.MAX_IMAGE_PIXELS = None

def is_valid_mask(mask, crop_size):
    if np.sum(mask) > crop_size ** 2 / 4:
        return False
    if mask.shape[0] < crop_size or mask.shape[1] < crop_size:
        return False
    return True

# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/dot"
# target_symbol = "16"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/drill_hole"
# target_symbol = "12"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/foliation_vertical"
# target_symbol = "27"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/inclined_bedding"
# target_symbol = "14"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/inclined_bedding"
# target_symbol = "44"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/inclined_bedding_with_top_direction"
# target_symbol = "46"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/inclined_flow_banding"
# target_symbol = "18"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/inclined_metamorphic"
# target_symbol = "19"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/lineation"
# target_symbol = "0"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/mine_shaft"
# target_symbol = "50"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/mine_tunnel"
# target_symbol = "43"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/overturned_bedding"
# target_symbol = "41"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/overturned_bedding_with_top_direction"
# target_symbol = "42"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/quarry"
# target_symbol = "39"
# target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/vertical_bedding"
# target_symbol = "35"
target_symbol_image_dir = "/Users/dong_dong_dong/Downloads/Darpa/fewshot/symbol_data/cropped_point_symbols/draft/vertical_bedding_with_top_direction"
target_symbol = "37"
# SYMBOL_BASE_SIZE = [(30, 30), (60, 60)]
SYMBOL_BASE_SIZE = [(60, 60)]

def generate_sythentic_data(target_symbol_image_dir, target_symbol, SYMBOL_BASE_SIZE, sythentic_data_path, max_num_synthetic_images, map_tif_dir, map_mask_dir):
    root1 = sythentic_data_path #output directory
    max_rotate = 360
    crop_size = 1024
    shift_size = 512

#    map_tif_dir = os.path.join("/Users/dong_dong_dong/Downloads/Darpa/train", 'train_input')  # The folder contains raw .tif and .json
    annotation_tif_dir = os.path.join(root1, 'train_output') # The folder contains raster layer

    # The folder contains legend image

#    map_mask_dir = "/Users/dong_dong_dong/Downloads/Darpa/train/intermediate6/cropped_map_mask"

    output_dir = os.path.join(root1, f'point_synthetic_maps/{target_symbol}') # folder for the final generated images and labels
    output_json_file = os.path.join(output_dir, 'train_poly.json')
    output_image_dir = os.path.join(output_dir, 'images')
    number_img_dir = "/Users/dong_dong_dong/Downloads/Darpa/criticalmaas-TA1-synthmap-points/symbol_data/numbers"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_image_dir)

    target_symbol_images = load_symbol_images(target_symbol_image_dir)
    all_tif_files = sorted(glob.glob(os.path.join(map_tif_dir, '*.tif')))

    output_json = {'images': [], 'annotations': [], 'img2anno': {}}
    max_count = 5
    allow_collision = False
    require_hollow = False

    image_id = 0
    for map_tif_idx, map_tif_file in enumerate(all_tif_files):

        map_name = map_tif_file.split('/')[-1].split('.')[0]
        print(f'>>> Processing candidate map tif: {map_name} ')
        start_time = time.time()

        map_img = Image.open(map_tif_file)
        map_img = np.array(map_img)
        print(map_mask_dir)
        print(map_name)
        map_mask = build_map_mask(map_name, map_mask_dir, map_tif_dir)
        try:
            point_mask = build_point_mask(map_name, map_img, annotation_tif_dir)
        except Exception as e:
            continue

        xs, ys = np.where(map_mask)
        roi_min_x, roi_max_x = np.min(xs), np.max(xs)
        roi_min_y, roi_max_y = np.min(ys), np.max(ys)

        map_index = 0
        for idx in range(roi_min_y, roi_max_y, shift_size):
            for jdx in range(roi_min_x, roi_max_x, shift_size):

                # skip the patch if there are no place to put target symbol
                map_mask_clip = map_mask[idx: idx + crop_size, jdx: jdx + crop_size]
                if not is_valid_mask(map_mask_clip, crop_size):
                    continue

                map_img_clip = map_img[idx: idx + crop_size, jdx: jdx + crop_size]

                # random load target symbol
                symbol_img = sample_symbol_image(target_symbol_images, SYMBOL_BASE_SIZE)
                sample_symbol_images = create_symbol_images(symbol_img,
                                                            max_rotate=max_rotate,
                                                            add_num_img=False,
                                                            number_img_dir=number_img_dir)

                output, valid_placements = synthesize(sample_symbol_images,
                                                      map_img_clip,
                                                      map_mask_clip,
                                                      max_count=max_count,
                                                      allow_collision=allow_collision,
                                                      require_hollow=require_hollow)

                # output
                output_image_filename = '{:07d}.jpg'.format(image_id)
                cv2.imwrite(os.path.join(output_image_dir, output_image_filename), output)
                image_id = int(output_image_filename.split('.')[0])
                labels_folder = os.path.join(output_dir, 'labels')
                if not os.path.exists(labels_folder):
                    os.makedirs(labels_folder)
                txt_file_path = os.path.join(labels_folder,f"{image_id:07d}.txt")
                with open(txt_file_path, 'w') as txt_file:
                    for (x, y, rotated_bbox, degree) in valid_placements:
                        x1, y1, x2, y2, x3, y3, x4, y4 = rotated_bbox.reshape(-1).tolist()[:-2]
                        height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
                        width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
                        # Calculate normalized cx, cy, width, and height
                        normalized_cx = x / crop_size
                        normalized_cy = y / crop_size
                        normalized_width = width / crop_size
                        normalized_height = height / crop_size

                        # Write normalized information to the text file
                        txt_file.write(
                            f"{target_symbol} {normalized_cx} {normalized_cy} {normalized_width} {normalized_height}\n")
                image_id += 1
                map_index += 1
                if map_index > max_num_synthetic_images:
                    break
            if map_index > max_num_synthetic_images:
                break