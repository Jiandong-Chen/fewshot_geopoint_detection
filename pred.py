import yaml
from pred_stitch_point_updated import *
from map2patch import crop_map_main

# map_input_dir_root = '/Users/dong_dong_dong/Downloads/Darpa/fewshot/sythentic_pred/point_synthetic_maps/NV_OutlawSprings_319708_1980_24000_geo_mosaic/images/test'
# model_weights_dir = '/Users/dong_dong_dong/Downloads/Darpa/fewshot/runs/detect'
# selected_model_weights = 'best.pt'
# selected_descriptions = 'NV_OutlawSprings_319708_1980_24000_geo_mosaic'
# predict_output_dir = '/Users/dong_dong_dong/Downloads/Darpa/fewshot/pred_res'
# output_dir_root = '/Users/dong_dong_dong/Downloads/Darpa/fewshot'
#
#
# folder_path = "/Users/dong_dong_dong/Downloads/Darpa/eval/eval_data_perfomer"
# generated_meta_info = '/Users/dong_dong_dong/Downloads/Darpa/legend_item_description_outputs/evaluation'
# map_patch_output_dir = '/Users/dong_dong_dong/Downloads/Darpa/cropped_maps'

def fewshot_prediction(model_weights_dir, rawmap_path, map_patch_output_dir, predict_output_dir, final_output_dir_root):
    for model in os.listdir(model_weights_dir):
        if not model.startswith('.'):
            model_path = os.path.join(model_weights_dir, model)
            model_weight = os.path.join(model_path, 'weights', 'best.pt')
            model_yaml = os.path.join(model_path, 'args.yaml')
            # Open the YAML file
            with open(model_yaml, 'r') as file:
                # Load YAML data
                data = yaml.safe_load(file)
            mapname = data['data'].split('/')[-2]
            map_path = os.path.join(rawmap_path, mapname + '.tif')
            crop_map_main(map_path, 1024, 1024, map_patch_output_dir)
            crop_dir_path = os.path.join(map_patch_output_dir, mapname + '_g1024_s1024')
            print(crop_dir_path)
            print(model_weight)
            predict_img_patches(crop_dir_path, model_weight, predict_output_dir)

            stitch_output_dir = os.path.join(final_output_dir_root, 'stitch')
            if not os.path.isdir(stitch_output_dir):
                os.mkdir(stitch_output_dir)

            print("=== Running a stitching module ===")
            stitch_to_single_result(mapname, predict_output_dir, stitch_output_dir)



# for img_name in os.listdir(folder_path):
#     if folder_path != ".DS_Store" and img_name.endswith('.tif'):
#         mapname = img_name.split('.')[0]
#         map_path = os.path.join(folder_path, img_name)
#         crop_map_main(map_path,1024, 1024, map_patch_output_dir)
#         for crop_map_folder in os.listdir(map_patch_output_dir):
#             if crop_map_folder != ".DS_Store":
#                 crop_map_path = os.path.join(map_patch_output_dir, crop_map_folder)
#                 for crop_map_patch in os.listdir(crop_map_path):
#                     if crop_map_folder != ".DS_Store":
#                         patch = os.path.join(crop_map_path, crop_map_patch)
#                         predict_img_patches(crop_dir_path, model_dir_root, selected_models, selected_descriptions,predict_output_dir)
#
