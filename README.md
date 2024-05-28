# Fewshot Point Detection
========================

This project developed an automatic pipeline for geo point symbol detection. A baseline model, yolov8n.pt, was trained on the competition training maps as well as synthetic point symbols. The entire competition validation dataset was used as the validation set. The pipeline performs few-shot detection by continuing train on the baseline model, generating a model for each prediction map using the synthetic point symbols in the map. This specifically trained model is then used to make predictions for the map.

Table of Contents
-----------------
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Contributing](#contributing)

Installation
------------
Python Version: 3.12

Package Version:
ultralytics 8.2.23
pillow 10.3.0
opencv-python 4.9.0.80
yaml 0.2.5
matplotlib 3.9.0
pandas 2.2.2
numpy 1.26.4
geojson 3.1.0
logging 0.4.9.6

Usage
-----
To run the script, navigate to the directory containing the main.py script and execute it using Python


Arguments
---------
The script needs the following arguments:

- `--rawmap_path`: Path of a list of prediction maps.
- `--main_path`: Main path for the project.
- `--generated_meta_info`: Path to the generated metainfo that contains map area coordinates.
- `--sythentic_data_output_path`: Path to the generated synthetic data.
- `--map_patch_output_dir`: Directory for cropped map patch outputs.
- `--predict_output_dir`: Directory for prediction outputs.
- `--final_output_dir_root`: Root directory for final outputs.
- `--baseline_model_path`: Path to the baseline model weights.

Contributing
------------
If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome all contributions.


