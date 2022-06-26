#!/usr/bin/env python
# coding: utf-8

# ## MIST Example Notebook

# #### Getting started
# 
# It is highly recommended that you run MIST in a Docker container with TensorFlow 2.6.0 or later. Once you start this Jupyter notebook in a container, uncomment the cell below to install some necessary dependencies.


#get_ipython().system('pip3 install pynvml')


# !conda install -U antspyx

# !conda install -U SimpleITK
# !conda install -U tqdm
# !conda install -U psutil
# !conda install -U tensorflow_addons


# #### Import the necessary scripts
# 
# The python file 'runtime.py' is the main workhorse for this pipeline. It calls the preprocessing pipeline, network architecture, and prediction metrics used for the global training and inference pipeline.


import json
import os
import sys 
sys.path.insert(1, '/rsrch1/ip/rglenn1/segLiverWkspce/MIST/') 
sys.path.insert(1, '/rsrch1/ip/rglenn1/segLiverWkspce/MIST/mist') 

# Don't show all of the tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Select your GPU (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Import runtime script
from mist.runtime import RunTime


# #### Create your user inputs file
# 
# To initiate the MIST pipeline, you need to provide it a JSON file with some parameters (i.e., path to data, output paths, and basic training parameters). Below is an example input JSON file to run the MIST pipeline on the LiTS dataset.


#!head /rsrch1/ip/rglenn1/data/trainingdata.csv
import pandas as pd
#! ls /rsrch4/ip/dtfuentes/github/hccdetection/anonymize



data_df = pd.read_csv('/rsrch1/ip/rglenn1/data/trainingdata.csv')

data_df = pd.read_csv('/rsrch1/ip/rglenn1/data/paths.csv')


# print(data_df)
# print(data_df['dataid'][0:10])
# print(data_df['uid'][0:10])
# print(data_df['image'][0:10])
# print(data_df['label'][0:10])
# print(data_df['train'][0:10])
# print(data_df['pre'][0:10])
# print(len(data_df['ven']))


user_params_lits = {'raw_data_dir': '/rsrch1/ip/rglenn1/data/Processed',
                    'processed_data_dir': '/rsrch1/ip/rglenn1/data/TFRecord',
                    'base_model_name': 'unet_pocket_detection',
                    'model_dir': '/rsrch1/ip/rglenn1/data/models',
                    'prediction_dir': '/rsrch1/ip/rglenn1/data/predictions',
                    'raw_paths_csv': '/rsrch1/ip/rglenn1/data/paths.csv',
                    'inferred_params': '/rsrch1/ip/rglenn1/data/hcc_params.json',
                    'results_path': '/rsrch1/ip/rglenn1/data/results',
                    'modality': 'mr',
                    'mask': ['Truth.raw.nii.gz'], 
                    'images': {'Art': ['Art.raw.nii.gz'],'Pre': ['Pre.raw.nii.gz'],'Ven': ['Ven.raw.nii.gz']}, 
                    'labels': [0, 1],
                    'final_classes': {'Liver': [1]},
                    'loss': 'dice', 
                    'model': 'unet', 
                    'pocket': True,
                    'gpu' :4}


# user_params_lits = {'raw_data_dir': '/tf/data/lits/raw/train',
#                     'processed_data_dir': '/tf/data/lits/processed/tfrecord',
#                     'base_model_name': 'mist_lits_example',
#                     'model_dir': '/tf/data/lits/models/mist_example',
#                     'prediction_dir': '/tf/data/lits/predictions/mist_example',
#                     'raw_paths_csv': '/tf/github/MIST/mist_example/lits_paths.csv',
#                     'inferred_params': '/tf/github/MIST/mist_example/lits_inferred_params.json',
#                     'results_csv': '/tf/github/MIST/mist_example/lits_results.csv',
#                     'modality': 'ct',
#                     'mask': ['segmentation'], 
#                     'images': {'volume': ['volume']}, 
#                     'labels': [0, 1, 2],
#                     'final_classes': {'Liver': [1, 2], 'Tumor': [2]},
#                     'loss': 'gdl', 
#                     'model': 'unet', 
#                     'pocket': True}

json_file = '/rsrch1/ip/rglenn1/segLiverWkspce/MIST/lits_user_params.json'
with open(json_file, 'w') as outfile: 
    json.dump(user_params_lits, outfile)


# #### Run MIST pipeline
# 
# Once you have your input JSON file, simply run command in the following cell to initiate the MIST training and inference pipeline. Enjoy!


# Create runtime instance
train = RunTime(json_file)

# Run the runtime instance
train.run(run_preprocess = False)


# #### Quick note about the ```run_preprocess``` parameter
# 
# The preprocessing pipeline can take quite a while to run depending on the training data. When running the MIST pipeline for the first time, set ```run_preprocess``` to ```True```. However, if you want to try a different 'model' or 'loss' parameter in your input JSON file after your initial run, then set the ```run_preprocess``` to ```False```. This will skip the preprocessing pipeline
