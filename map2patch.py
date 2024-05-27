import time
import os
import cv2
import glob
import json
import numpy as np

def crop_map(map_image, map_name, patch_size, stride, output_dir):

    h, w = map_image.shape[:2]
    p_h, p_w = patch_size, patch_size
    if h % stride == 0:
        num_h = h // stride
    else:
        num_h = h // stride + 1

    if w % stride == 0:
        num_w = w // stride
    else:
        num_w = w // stride + 1

    output_folder = os.path.join(output_dir, f'{map_name}_g{patch_size}_s{stride}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    x_ = np.array([i for i in range(0, patch_size * num_h, stride)])
    y_ = np.array([i for i in range(0, patch_size * num_w, stride)])

    ind = np.meshgrid(x_, y_, indexing='ij')

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
        patch = map_image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :]

        output_path = os.path.join(output_folder, f'{map_name}_{start[0]}_{start[1]}.png')
        cv2.imwrite(output_path, patch)

    return output_folder

def crop_map_main(map_path, patch_size, stride, output_dir):
    input_path = map_path
    map_name = os.path.basename(input_path).split('.')[0]
    map_image = cv2.imread(input_path)

    print(f'*** generating {patch_size} for {os.path.basename(map_path)} ***')
    s_time = time.time()

    output_path = crop_map(map_image, map_name, patch_size, stride, output_dir)

    e_time = time.time()
    print(f'processing time {e_time-s_time}s')
    print(f'*** saved the cropped images for {map_name} in {output_path}')
        
        
