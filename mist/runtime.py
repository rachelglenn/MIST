import os
import gc
import json
import ants
import random
import psutil
import scipy
import pandas as pd
import numpy as np
from tqdm import trange
import signal
import time
import readchar

from mist.kFoldMetrics import kFoldMetrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.ndimage.filters import gaussian_filter
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler # This function keeps the learning rate at 0.001 for the first ten epochs


from mist.model import *
from mist.loss import *
from mist.preprocess import *
from mist.metrics import *
from mist.utils import *
from mist.kFoldCallback import *


import warnings
warnings.simplefilter(action = 'ignore', 
                      category = np.VisibleDeprecationWarning)

warnings.simplefilter(action = 'ignore', 
                      category = FutureWarning)


from scipy import ndimage

def downsamplePatient(patient_CT, resize_factor):
    original_CT = sitk.ReadImage(patient_CT,sitk.sitkInt32)
    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
  

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    
    return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)

class RunTime(object):
    
    def __init__(self, json_file):
        # Read user defined parameters
        with open(json_file, 'r') as file:
            self.params = json.load(file)
        self.json_file = json_file                
        # Get loss function and preprocessor
        self.loss = Loss(json_file)
        self.preprocess = Preprocess(json_file)

        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
        self.n_folds = 5
        self.epochs =  100
        self.steps_per_epoch = 250
        self.k_metrics = None


    def handler(self, signum, frame):
        msg = "Ctrl-c was pressed. Do you want to save the progress? y/n "
        print(msg, end="", flush=True)
        res = readchar.readchar()
        if res == 'y':
            print("")
            if self.k_metrics != None:

                self.k_metrics.on_training_end()
            exit(1)
        else:
            exit(1)

    def decode(self, serialized_example):
        features_dict = {'image': tf.io.VarLenFeature(tf.float32),
                         'mask': tf.io.VarLenFeature(tf.float32),
                         'dims': tf.io.FixedLenFeature([3], tf.int64),
                         'num_channels': tf.io.FixedLenFeature([1], tf.int64), 
                         'num_classes': tf.io.FixedLenFeature([1], tf.int64), 
                         'label_points': tf.io.VarLenFeature(tf.int64), 
                         'label_index_ranges': tf.io.FixedLenFeature([len(self.params['labels']) + 1], tf.int64)}
        
        # Decode examples stored in TFRecord
        features = tf.io.parse_example(serialized_example, features_dict)
        
        # Crop random patch from images
        # Extract image/mask pair from sparse tensors
        image = tf.sparse.to_dense(features['image'])
        image = tf.reshape(image, tf.concat([features['dims'], features['num_channels']], axis = -1))

        mask = tf.sparse.to_dense(features['mask'])
        mask = tf.reshape(mask, tf.concat([features['dims'], features['num_classes']], axis = -1))
        
        # Get image dimensions
        dims = features['dims']
        num_channels = features['num_channels']
        num_classes = features['num_classes']

        # Extract point lists for each label
        # TF constant for reshaping list of points to array
        three = tf.constant(3, shape = (1,), dtype = tf.int64)
        label_index_ranges = features['label_index_ranges']
        #print('label_index_range', [len(self.params['labels']) + 1])
        num_points = tf.reshape(label_index_ranges[-1], shape = (1,))
        label_points = tf.sparse.to_dense(features['label_points'])
        label_points = tf.reshape(label_points, tf.concat([three, num_points], axis = -1))
        
        return image, mask, dims, num_channels, num_classes, label_points, label_index_ranges
    
    def decode_val(self, serialized_example):
        features_dict = {'image': tf.io.VarLenFeature(tf.float32),
                         'mask': tf.io.VarLenFeature(tf.float32),
                         'dims': tf.io.FixedLenFeature([3], tf.int64),
                         'num_channels': tf.io.FixedLenFeature([1], tf.int64), 
                         'num_classes': tf.io.FixedLenFeature([1], tf.int64), 
                         'label_points': tf.io.VarLenFeature(tf.int64), 
                         'label_index_ranges': tf.io.FixedLenFeature([len(self.params['labels']) + 1], tf.int64)}
        
        # Decode examples stored in TFRecord
        features = tf.io.parse_example(serialized_example, features_dict)
        
        # Extract image/mask pair from sparse tensors
        image = tf.sparse.to_dense(features['image'])
        image = tf.reshape(image, tf.concat([features['dims'], features['num_channels']], axis = -1))
        # print("Image Size:", image.shape)

        mask = tf.sparse.to_dense(features['mask'])
        mask = tf.reshape(mask, tf.concat([features['dims'], features['num_classes']], axis = -1))
        
        return image, mask

    def random_crop(self, image, mask, dims, num_channels, num_classes, label_points, label_index_ranges, fg_prob):
        if tf.random.uniform([]) <= fg_prob:
            # Pick a foreground point (i.e., any label that is not 0)
            # Randomly pick a foreground class
            print('check_min_max 176',1,  len(self.params['labels']))
            label_idx = tf.random.uniform([], 
                                          minval = 1, 
                                          maxval = len(self.params['labels']), 
                                          dtype = tf.int32)
            low = label_idx
            high = label_idx + 1
            
            # If the label is not in the image, then pick any foreground label
            if label_index_ranges[high] <= label_index_ranges[low]:
                low = 1
                high = -1
        else:
            low = 0
            high = 1
            
        # Pick center point for patch
        print('check_min_max 192', low, high ) # For brats, 3 values Might need to change with VF
        # point_idx = tf.random.uniform([], 
        #                               minval = label_index_ranges[low], 
        #                               maxval = label_index_ranges[high], 
        #                               dtype=tf.int64)
        
        # RG found bug! Temporary fix
        point_idx = tf.random.uniform([], 
                                      minval = 0, 
                                      maxval = 1, 
                                      dtype=tf.int64)
        point = label_points[..., point_idx]
            
        # Extract random patch from image/mask
        patch_radius = [patch_dim // 2 for patch_dim in self.params['patch_size']]
        padding_x = self.params['patch_size'][0] - (tf.reduce_min([dims[0], point[0] + patch_radius[0]]) - tf.reduce_max([0, point[0] - patch_radius[0]]))
        padding_y = self.params['patch_size'][1] - (tf.reduce_min([dims[1], point[1] + patch_radius[1]]) - tf.reduce_max([0, point[1] - patch_radius[1]]))
        padding_z = self.params['patch_size'][2] - (tf.reduce_min([dims[2], point[2] + patch_radius[2]]) - tf.reduce_max([0, point[2] - patch_radius[2]]))
        
        zero = tf.constant(0, tf.int64)
        two = tf.constant(2, tf.int64)
        one = tf.constant(1, tf.int64)
        if tf.math.floormod(padding_x, two) > zero:
            padding_x = tf.stack([padding_x // 2, (padding_x // 2) + one])
        else:
            padding_x = tf.stack([padding_x // 2, padding_x // 2])
            
        if tf.math.floormod(padding_y, two) > zero:
            padding_y = tf.stack([padding_y // 2, (padding_y // 2) + one])
        else:
            padding_y = tf.stack([padding_y // 2, padding_y // 2])
            
        if tf.math.floormod(padding_z, two) > zero:
            padding_z = tf.stack([padding_z // 2, (padding_z // 2) + one])
        else:
            padding_z = tf.stack([padding_z // 2, padding_z // 2])

        padding_c = tf.stack([zero, zero])
        padding = tf.stack([padding_x, padding_y, padding_z, padding_c])
        
        image_patch = image[tf.reduce_max([0, point[0] - patch_radius[0]]):tf.reduce_min([dims[0], point[0] + patch_radius[0]]), 
                            tf.reduce_max([0, point[1] - patch_radius[1]]):tf.reduce_min([dims[1], point[1] + patch_radius[1]]), 
                            tf.reduce_max([0, point[2] - patch_radius[2]]):tf.reduce_min([dims[2], point[2] + patch_radius[2]]), ...]
        
        image_patch = tf.pad(image_patch, padding)
                
        mask_patch = mask[tf.reduce_max([0, point[0] - patch_radius[0]]):tf.reduce_min([dims[0], point[0] + patch_radius[0]]), 
                          tf.reduce_max([0, point[1] - patch_radius[1]]):tf.reduce_min([dims[1], point[1] + patch_radius[1]]), 
                          tf.reduce_max([0, point[2] - patch_radius[2]]):tf.reduce_min([dims[2], point[2] + patch_radius[2]]), ...]
        
        mask_patch = tf.pad(mask_patch, padding)
                
        # Random augmentation
        # Random flips
        if tf.random.uniform([]) <= 0.15:
            axis = np.random.randint(0, 3)
            if axis == 0:
                image_patch = image_patch[::-1, :, :, ...]
                mask_patch = mask_patch[::-1, :, :, ...]
            elif axis == 1:
                image_patch = image_patch[:, ::-1, :, ...]
                mask_patch = mask_patch[:, ::-1, :, ...]
            else:
                image_patch = image_patch[:, :, ::-1, ...]
                mask_patch = mask_patch[:, :, ::-1, ...]

        # Random noise
        if tf.random.uniform([]) <= 0.15:
            variance = tf.random.uniform([], minval = 0.001, maxval = 0.05)
            
            if self.params['modality'] == 'mr':
                # Add Rician noise if using MR images
                image_patch = tf.math.sqrt(
                    tf.math.square((image_patch + tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) + 
                    tf.math.square(tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) * tf.math.sign(image_patch)
            else:
                # Otherwise, use Gaussian noise
                image_patch += tf.random.normal(shape = tf.shape(image_patch), stddev = variance)

        # Apply Gaussian blur to image
        if tf.random.uniform([]) <= 0.15:
            # TODO: Apply Gaussian noise to random channels
            blur_level = np.random.uniform(0.25, 0.75)
            image_patch = tf.numpy_function(scipy.ndimage.gaussian_filter, [image_patch, blur_level], tf.float32)
                    
        return image_patch, mask_patch
 
    @timeit
    def testSet(self, tfrecords, split):
        test_tfr_list = [tfrecords[idx] for idx in split]
        test_df_ids = [self.df.iloc[idx]['id'] for idx in split]
        test_df = self.df.loc[self.df['id'].isin(test_df_ids)].reset_index(drop = True)
        
        # Run prediction on test set and write results to .nii.gz format
        test_ds = tf.data.TFRecordDataset(test_tfr_list, 
                                            compression_type = 'GZIP', 
                                            num_parallel_reads = tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.map(self.decode_val, num_parallel_calls = tf.data.experimental.AUTOTUNE)

        filename = self.params['results_path'] + "/" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
        filename += "_testset.txt"
        np.savetxt(filename, np.asarray(test_df_ids), fmt = '%s', delimiter='\n')
        return test_df, test_ds


    def trainingValidationSet(self, train_cache, cache_size, crop_fn, val_tfr_list):
        # Prepare training set
        train_ds = tf.data.TFRecordDataset(train_cache, 
                                            compression_type = 'GZIP', 
                                            num_parallel_reads = tf.data.experimental.AUTOTUNE)

        if cache_size < 5:
            train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        else:
            train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.experimental.AUTOTUNE).cache()

        train_ds = train_ds.map(crop_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.batch(batch_size = 2, drop_remainder = True)
        train_ds = train_ds.repeat()
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        val_ds = tf.data.TFRecordDataset(val_tfr_list, 
                                            compression_type = 'GZIP', 
                                            num_parallel_reads = tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(self.decode_val, num_parallel_calls = tf.data.experimental.AUTOTUNE)

        filename =  self.params['results_path'] + "/" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
        filename += "_trainset.txt"
        # print("train_cache", train_cache)
        np.savetxt(filename, np.array(train_cache),  fmt = '%s',  delimiter='\n')
        
        filename =  self.params['results_path'] + "/" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
        filename += "_validset.txt"
        np.savetxt(filename, np.asarray(val_tfr_list),  fmt = '%s',  delimiter='\n')

        return train_ds, val_ds

            
    def alpha_schedule(self, step): 
        #TODO: Make a step-function scheduler and an adaptive option
        return (-1. / float(self.epochs)) * float(step) + 1
    
    def train(self):

        # Get folds for k-fold cross validation 42, 24, 12, 9, 4, 11, 34, 56, 34, 2) 
        kfold = KFold(n_splits = self.n_folds, shuffle = True, random_state = 11)

        # Get Training data
        self.df = pd.read_csv(self.params['raw_paths_csv'])
        
        # Convert to tfrecord
        tfrecords = [os.path.join(self.params['processed_data_dir'], 
                                  '{}.tfrecord'.format(self.df.iloc[i]['id'])) for i in range(len(self.df))]
        tfrecords = sorted(tfrecords)


        print("tfrecords", len(tfrecords), self.params['processed_data_dir'])
        #tfrecords1 = tfrecords[1:5]
        #tfrecords = tfrecords[1:17] + tfrecords[25:30]
        #split the data
        splits = kfold.split(tfrecords)
        
        # Extract folds so that users can specify folds to train on
        train_splits = list()
        test_splits = list()
        for split in splits:
            train_splits.append(sorted(split[0]))
            test_splits.append(sorted(split[1]))

        train_splits = train_splits
        test_splits = test_splits

        # Setup Model
        depth, cache_size = self.inferredParams()
 
        # Oversample patches centered at foreground voxels
        fg_prob = 0.85
        crop_fn = lambda image, mask, dims, num_channels, num_classes, label_points, label_index_ranges: self.random_crop(image,
                                                                                                                          mask,
                                                                                                                          dims,
                                                                                                                          num_channels,
                                                                                                                          num_classes,
                                                                                                                     label_points,
                                                                                                                          label_index_ranges,
                                                                                                                          fg_prob)
        if self.multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = None
        
        # Setup Model metrics for kFold training
        self.k_metrics = kFoldMetrics(self.params)

        ### Start training loop ###
        split_cnt = 1
        print("Folds:", self.params['folds'])
        for fold in self.params['folds']:
            print('Starting fold {}...'.format(fold))
            train_tfr_list = [tfrecords[idx] for idx in train_splits[fold]]
            

            # Get test dataset 
            test_df, test_ds = self.testSet(tfrecords, test_splits[fold])

            if 0.1*len(train_tfr_list) >= 10:
                test_size = 10./len(train_tfr_list)
            else:
                test_size = 0.1
                
            # Get validation tfrecords from training split
            train_tfr_list, val_tfr_list, _, _ = train_test_split(train_tfr_list,
                                                                  train_tfr_list,
                                                                  test_size = test_size,
                                                                  random_state = 42)
            #df = pd.DataFrame(train_tfr_list)
            #np.random.shuffle(train_tfr_list)
            #training, test = x[:80,:], x[80:,:]
            #train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
            #train_tfr_list, val_tfr_list = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
            
            if cache_size < len(train_tfr_list):
                # Initialize training cache and pool in first epoch
                train_cache = random.sample(train_tfr_list, cache_size)
                random.shuffle(train_cache)
                cache_pool = list(set(train_tfr_list) - set(train_cache))
                random.shuffle(cache_pool)
            else:
                train_cache = train_tfr_list

            # Prepare training and validation set
            train_ds, val_ds = self.trainingValidationSet(train_cache, cache_size, crop_fn, sorted(val_tfr_list))
            print('check size', np.shape(train_ds))
            print('check size', np.shape(val_ds))
        
            # ####RG changed to numpy files
            # processed_data_folder = '/rsrch1/ip/rglenn1/data/VF_datset'

            # images_dir = os.path.join(processed_data_folder, 'images')
            # images = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
            # #images = self.df['image']
            # #print(images)
            

            # labels_dir = os.path.join(processed_data_folder, 'labels')
            # labels = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir)]
            # #labels = self.df['mask']
            # splits = kfold.split(list(range(len(images))))

            #         # Extract folds so that users can specify folds to train on
            # train_splits = list()
            # test_splits = list()
            # for split in splits:
            #     train_splits.append(split[0])
            #     test_splits.append(split[1])
            # train_images = [images[idx] for idx in train_splits[fold]]
            # train_labels = [labels[idx] for idx in train_splits[fold]]
            
            # test_images = [images[idx] for idx in test_splits[fold]]
            # test_images.sort()

            # test_labels = [labels[idx] for idx in test_splits[fold]]
            # test_labels.sort()

            
            # # Get validation set from training split
            # train_images, val_images, train_labels, val_labels = train_test_split(train_images,
            #                                                                       train_labels,
            #                                                                       test_size=0.1,
            #                                                                       random_state=42)
                                                                                  
            # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            # BATCH_SIZE = 2
            # SHUFFLE_BUFFER_SIZE = 10

            # train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
            # test_dataset = test_dataset.batch(BATCH_SIZE)
            # ###############End RG



            for i in range(self.epochs):
                print('Epoch {}/{}'.format(i + 1, self.epochs))
               
                learningrate = self.k_metrics.learningrate
                model = self.setupModel( i, learningrate, depth, strategy)   
                from keras.utils.vis_utils import plot_model
                plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
                #model.summary()

                # Setup tensorboard
                idfold = split_cnt
                #os.system('mkdir -p log/%d/' % idfold)
                #tensorboard = TensorBoard(log_dir='./log/%d/' % idfold, histogram_freq=0, write_graph=True, write_images=False)
    


               
                # Train model
                training_history = model.fit(train_ds, #RG change to numpy files
                          epochs = 1, 
                          steps_per_epoch =  self.steps_per_epoch)
                          #  steps_per_epoch =  len(train_cache))
                          #steps_per_epoch = (len(train_cache)) // batchSize)
                          #validation_data = validationGenerator ,
                          #validation_steps = (len(val)) // batchSize,
                          #callbacks = [
                          #kFoldCallback(self.params, test_ds, test_df, val_ds, self.loss, self.epochs)])
                # Save model for next epoch

                self.k_metrics.on_epoch_end(model, val_ds, self.loss)

                # model.save(self.params['current_model_name'])
                
                if cache_size < len(train_tfr_list):
                    print("Change patient list...")
                    cache_replacement_rate = 0.2
                    # Pick n_replacement new patients from pool and remove the same number from the current cache
                    n_replacements = int(np.ceil(cache_size * cache_replacement_rate))
                    
                    # Bug fix: If n_replacements if greater than size of pool, use the entire pool
                    # This happens when the cache_size is large
                    n_replacements = int(np.min([len(cache_pool), n_replacements]))
                    new_cache_patients = random.sample(cache_pool, n_replacements)
                    back_to_pool_patients = random.sample(train_cache, n_replacements)

                    # Update cache and pool for next epoch
                    train_cache = list(set(train_cache) - set(back_to_pool_patients)) + new_cache_patients
                    random.shuffle(train_cache)
                    cache_pool = list(set(cache_pool) - set(new_cache_patients)) + back_to_pool_patients
                    random.shuffle(cache_pool)



                #d
                K.clear_session()
                gc.collect()

                ### End of epoch ###
            self.k_metrics.on_kFold_end( model, test_df, test_ds)
            del train_ds, val_ds, test_df, test_ds
            K.clear_session()
            gc.collect() 
        ### End of training ###
        
        self.k_metrics.on_training_end()
        del model
        K.clear_session()
        gc.collect()

        ### End train function ###

    def inferredParams(self ):
        #Get inferred_params
        filename = self.params['inferred_params']
            
        with open(filename, 'r') as file:
            inferred_params = json.load(file)
       
        # Setup Parameters
        loss_type = self.params['loss']
        labels = self.params['labels']
        modelname = self.params['model']
        pocket = self.params['pocket']
        df = self.df 
        n_channels = self.n_channels
        n_classes = self.n_classes
        

        #TODO understand how to get the median_image_size
        
        if 'median_image_size' in self.params.keys():
            median_image_size = inferred_params['median_image_size']
            available_mem = psutil.virtual_memory().available
            if (loss_type == 'dice') or (loss_type == 'gdl'):
                image_buffer_size = 4 * (np.prod(median_image_size) * (n_channels + len(labels)))
            else:
                image_buffer_size = 4 * (np.prod(median_image_size) * (n_channels + (2 * len(labels))))

            # Set cache size so that we do not exceed 2.5% of available memory
            cache_size = int(np.ceil((0.05 * available_mem) / image_buffer_size))
        else:
            
            print("Error: Median_image_size not found rerun with preprocessing")
            cache_size = 500*100
            median_image_size = [134, 170, 137] # 33 patient
            #median_image_size = [512, 512, 51] # RG VF
            #patch_size = [64,64,64] # RG VF
           


        if 'patch_size' in self.params.keys():
            patch_size = self.params['patch_size']
            depth = int(np.log(np.min(patch_size) / 4) / np.log(2))
        else:
            # Compute the biggest patch size that will fit on the system
            # Get candidate patch size from median image size
            patch_size = [get_nearest_power(median_image_size[i]) for i in range(3)]
            patch_size = [int(patch_size[i]) for i in range(3)]
            #patch_size = [64,64,64] # RG VF

            # Get available GPU memory
            gpu_memory_needed = np.Inf
            _, gpu_memory_available = auto_select_gpu()
            patch_reduction_switch = 1

            # Check if patch size fits on available memeory,
            # if it does not, then reduce the patch size until
            # it does
            while gpu_memory_needed >= gpu_memory_available:
                # Compute network depth based on patch size
                depth = int(np.log(np.min(patch_size) / 4) / np.log(2))

                # Build model from scratch in first epoch
                model = get_model(modelname, 
                                  patch_size = tuple(patch_size), 
                                  num_channels = n_channels,
                                  num_class = n_classes, 
                                  init_filters = 32, 
                                  depth = depth, 
                                  pocket = pocket)

                # Compute the GPU memory needed for candidate model
                gpu_memory_needed = get_model_memory_usage(2, model)

                # If needed memory exceeds available memory, then
                # reduce patch size and check new model
                if gpu_memory_needed > gpu_memory_available:
                    if patch_reduction_switch == 1:
                        patch_size[0] /= 2
                        patch_size[1] /= 2
                        patch_size = [int(patch_size[i]) for i in range(3)]
                        patch_reduction_switch = 2
                    else:
                        patch_size[2] /= 2
                        patch_size = [int(patch_size[i]) for i in range(3)]
                        patch_reduction_switch = 1

            ### End of while loop ###
            
            del model
            K.clear_session()
            gc.collect()
            
        inferred_params['patch_size'] = patch_size
        


        # Save inferred parameters as json file
        inferred_params_json_file = os.path.abspath(self.params['inferred_params'])
        with open(inferred_params_json_file, 'w') as outfile:
            json.dump(inferred_params, outfile)

        # Default case if folds are not specified by user
        if not('folds' in self.params.keys()):
            self.params['folds'] = [i for i in range(self.n_folds)]

        # Convert folds input to list if it is not already one
        if not(isinstance(self.params['folds'], list)):
            self.params['folds'] = [int(self.params['folds'])]

        #Merge inferred parameters with paramters
        self.params = merge_two_dicts(self.params, inferred_params)
        self.params['learning_rate'] = 0.001

        self.params['current_model_name'] =  os.path.join(self.params['model_dir'], '{}_current_model_split'.format(self.params['base_model_name']))
        self.params['best_model_name'] = os.path.join(self.params['model_dir'], '{}_best_model_split'.format(self.params['base_model_name']))
           
        
        return depth, cache_size

    def setupModel(self, step, learning_rate, depth, strategy):
        #epoch = 0 # doesn't matter for dice
        #alpha = self.alpha_schedule()
        #self.params['learning_rate'] = 0.001
        alpha = self.alpha_schedule(step)
        opt = tf.optimizers.Adam(learning_rate = learning_rate) # Change for VF
        #opt = tf.keras.optimizers.SGD(
        #    learning_rate=learning_rate,
        #    momentum=0.0,
        #    nesterov=False,
        #    clipnorm=None,
        #    clipvalue=None,
        #    global_clipnorm=None,
        #    name="SGD"
        #)
        if os.path.exists(self.params['best_model_name']):
            print("loading previous model...",self.params['best_model_name'])
            if self.multi_gpu and strategy != None:
                with strategy.scope():
                    # Reload model and resume training for later epochs
                    model = load_model(self.params['best_model_name'], custom_objects = {'loss': self.loss.loss_wrapper(alpha)})
                    # tf.config.optimizer.set_jit(True)
                    model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
            else:
                # Reload model and resume training for later epochs
                model = load_model(self.params['best_model_name'], custom_objects = {'loss': self.loss.loss_wrapper(alpha)})
                # tf.config.optimizer.set_jit(True)
                model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
        else:
            print("loading new model...")
            #if epoch == 0:
            if self.multi_gpu and strategy != None:
                with strategy.scope():
                    # Get model for this fold
                    model = get_model2(self.params['model'], 
                                        patch_size = tuple(self.params['patch_size']), 
                                        num_channels = self.n_channels,
                                        num_class = self.n_classes, 
                                        init_filters = 32, 
                                        depth = depth, 
                                        pocket = self.params['pocket'],
                                        json_file = self.json_file)
                    # tf.config.optimizer.set_jit(True)
                    model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
            else:
                # Get model for this fold
                model = get_model2(self.params['model'], 
                                    patch_size = tuple(self.params['patch_size']), 
                                    num_channels = self.n_channels,
                                    num_class = self.n_classes, 
                                    init_filters = 32, 
                                    depth = depth, 
                                    pocket = self.params['pocket'], 
                                    json_file = self.json_file)
                # tf.config.optimizer.set_jit(True)
                model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
        return model

    def setupGPU(self, gpu_specified):

        # Set HDF file locking to use model checkpoint
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

        if gpu_specified == 'auto':
            # Auto select GPU if using single GPU
            gpu_id, available_mem = auto_select_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            self.n_gpus = 1
            self.multi_gpu = False
        elif isinstance(gpu_specified, int):
            # Use user specified gpu
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_specified)
            self.n_gpus = 1
            self.multi_gpu = False
        elif isinstance(gpu_specified, list):
            if len(gpu_specified) > 1:
                self.multi_gpu = True
                self.n_gpus = len(gpu_specified)
                visible_devices = ''.join([str(gpu_specified[j]) for j in range(len(gpu_specified))])
                visible_devices = re.sub(r'([1-9])(?!$)', r'\1,', visible_devices)
                os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_specified[0])
                self.n_gpus = 1
                self.multi_gpu = False
        else:
            # If no gpu is specified, default to auto selection
            gpu_id, available_mem = auto_select_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            self.n_gpus = 1
            self.multi_gpu = False

        # Get GPUs
        gpus = tf.config.list_physical_devices('GPU')

        # Set mixed precision policy if compute capability >= 7.0
        use_mixed_policy = True
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            compute_capability = details['compute_capability'][0]
            if compute_capability < 7:
                use_mixed_policy = False
                break

        if use_mixed_policy:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            
        # For tensorflow 2.x.x allow memory growth on GPU
        ###################################
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        ###################################


    def run(self, run_preprocess = True):
        #print("Loss:", self.params['loss'])
        # Check if necessary directories exist, and creat them
        self.setupDir()


        # Set seed for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        if run_preprocess:
            # Preprocess data if running for the first time
            self.preprocess.run()
                        
        print('Setting up GPU...')
        # Select GPU for training
        # TODO: Make multi gpu training and option
        if 'gpu' in self.params.keys():
            self.setupGPU(self.params['gpu'])
        else:
            self.setupGPU(None)

        # Setup Signal handler
        signal.signal(signal.SIGINT, self.handler)
        
        #while True:
        # Run training pipeline
        self.train()


    def predict(self):
        # Get Training data
        self.df = pd.read_csv(self.params['raw_paths_csv'])
        
        # Convert to tfrecord
        tfrecords = [os.path.join(self.params['processed_data_dir'], 
                                  '{}.tfrecord'.format(self.df.iloc[i]['id'])) for i in range(len(self.df))]
    

        

    def setupDir(self):
         # Check if necessary directories exists, create them if not
        if not(os.path.exists(self.params['raw_data_dir'])):
            raise Exception('{} does not exist'.format(self.params['raw_data_dir']))
            
        if not(os.path.exists(self.params['processed_data_dir'])):
            os.mkdir(self.params['processed_data_dir'])
            
        if not(os.path.exists(self.params['model_dir'])):
            os.mkdir(self.params['model_dir'])
            
        if not(os.path.exists(self.params['prediction_dir'])):
            os.mkdir(self.params['prediction_dir'])

        if not(os.path.exists(os.path.join(self.params['prediction_dir'], 'raw'))):
            os.mkdir(os.path.join(self.params['prediction_dir'], 'raw'))
            
        if not(os.path.exists(os.path.join(self.params['prediction_dir'], 'postprocess'))):
            os.mkdir(os.path.join(self.params['prediction_dir'], 'postprocess'))
