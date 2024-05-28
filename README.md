# Fewshot Point Detection
========================

This project involves a script for parsing various paths required for a few-shot point detection task. The script uses the `argparse` library to handle input arguments for different file paths and directories.

Table of Contents
-----------------
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Contributing](#contributing)
- [License](#license)

Installation
------------
To use this script, you need to have Python installed on your machine. Additionally, make sure to install the `argparse` module, although it is included in the standard library for Python versions 2.7 and 3.2 and above.

Usage
-----
To run the script, navigate to the directory containing the script and execute it using Python:


Arguments
---------
The script accepts the following arguments:

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

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

