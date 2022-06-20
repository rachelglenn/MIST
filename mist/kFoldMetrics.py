import pandas as pd 
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
import ants
from scipy.ndimage.filters import gaussian_filter
from metrics import *
from datetime import datetime
import tensorflow.keras.backend as K
from mist.utils import timeit

class kFoldMetrics(Metrics):
    def __init__(self, params):
        
        # Model parameters
        self.learningrate = params['learning_rate']
        self.num_epochs = 0
        self.n_classes = len(params['labels'])
        self.patch_size = params['patch_size']
        self.dice = params['loss']
        self.best_val_loss = np.Inf
        self.plateau_cnt = 1
        self.use_nz_mask = params['use_nz_mask']
        self.target_spacing = params['target_spacing']
        self.min_component_size = params['min_component_size']
        self.prediction_dir = params['prediction_dir']
        #self.results_csv = params['results_csv']
        self.labels = params['labels']
        self.final_classes = params['final_classes']
        self.best_model_name = params['best_model_name']
        self.current_model_name = params['current_model_name']
        self.model_name = params['base_model_name']
        self.resultsPath = params['results_path']
        print('final_classes', self.final_classes)
        # Initialize results dataframe
        metrics = ['dice', 'haus95', 'avg_surf']
        results_cols = ['id']
        for metric in metrics: 
            for key in self.final_classes.keys():
                results_cols.append('{}_{}'.format(key, metric))
                
        self.results_df = pd.DataFrame(columns = results_cols)
        self.loss_end_of_epoch = []
        
        # Time metrics
        self.time_started = None
        self.time_finsihed = None
        self.on_train_begin()

        

    def on_train_begin(self):
        self.time_started = datetime.now()
        print(f'Training Started | {self.time_started}\n')


    def on_training_end(self):
        self.time_finished = datetime.now()
        train_duration = str(self.time_finished - self.time_started)
        print(f'\nTraining Finished | {self.time_finished} | Duration: {train_duration}')
      

        # Get final statistics
        self.performanceMetrics()

        self.plot_model_performance()
        
        self.plotMetrics()
        
        #print( f"Training loss:     {logs['loss']:.5f}")
    @timeit  
    def on_kFold_end(self, model, df, ds):
        self.val_inference(model, df, ds)
        self.plateau_cnt = 1
        self.best_val_loss = np.Inf
        self.learningrate = 0.001
    
    @timeit
    def on_epoch_end(self, model, val_ds, FuncLoss):
        self.num_epochs += 1

        # Comput loss for validation patients
        val_loss = self.patient_loss(model, val_ds, FuncLoss)
        
        if val_loss < self.best_val_loss:
            print('Val loss IMPROVED from {} to {}'.format(self.best_val_loss, val_loss))
            self.best_val_loss = val_loss

            model.save(self.best_model_name)
            self.plateau_cnt = 1

            K.clear_session()
            gc.collect()
        else:
            print('Val loss of DID NOT improve from {}'.format(self.best_val_loss))
           
            self.plateau_cnt += 1
        if self.plateau_cnt % 10:
            #self.model.optimizer.lr = self.model.optimizer.lr*0.9
            self.learningrate = self.learningrate*0.9
            self.plateau = 1
            print('Decreasing learning rate to {}'.format( self.learningrate))
        tl = val_loss
        # ta = logs['accuracy']
        # vl = logs['val_loss']
        # va = logs['val_accuracy']
        self.loss_end_of_epoch.append(tl)
        return self.learningrate

    @timeit
    def val_inference(self, model, df, ds):
        
        cnt = 0
        gaussian_map = self.get_gaussian()
                
        iterator = ds.as_numpy_iterator()
        for element in iterator:
            
            patient = df.iloc[cnt].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            
            original_mask = ants.image_read(patient['mask'])
            original_image = ants.image_read(image_list[0])
            original_dims = ants.image_header_info(image_list[0])['dimensions']
            
            if self.use_nz_mask:
                nzmask = ants.get_mask(original_image, cleanup = 0)
                original_cropped = ants.crop_image(original_mask, nzmask)
            
            image = element[0]
            truth = element[1]
            dims = image[..., 0].shape
                        
            padding = list()
            cropping = list()
            for i in range(3):
                if dims[i] % self.patch_size[i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / self.patch_size[i]) * self.patch_size[i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape
            
            strides = [patch_dim // 2 for patch_dim in self.patch_size]
            prediction = np.zeros((*pad_dims, self.n_classes))
            for i in range(0, pad_dims[0] - self.patch_size[0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.patch_size[1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.patch_size[2] + 1, strides[2]):
                        patch = image[i:(i + self.patch_size[0]),
                                        j:(j + self.patch_size[1]),
                                        k:(k + self.patch_size[2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        pred_patch = model.predict(patch)
                        pred_patch *= gaussian_map
                        prediction[i:(i + self.patch_size[0]),
                                    j:(j + self.patch_size[1]),
                                    k:(k + self.patch_size[2]), ...] = pred_patch
            
            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]

            prediction = np.argmax(prediction, axis = -1)
            
            # Make sure that labels are correct in prediction
            for j in range(self.n_classes):
                prediction[prediction == j] = self.labels[j]
                
            prediction = prediction.astype('float32')

            if self.use_nz_mask:
                prediction = original_cropped.new_image_like(data = prediction)
                prediction = ants.decrop_image(prediction, original_mask)
            else:
                # Put prediction back into original image space
                prediction = ants.from_numpy(prediction)
                
            prediction.set_spacing(self.target_spacing)
                        
            if np.linalg.norm(np.array(prediction.direction) - np.eye(3)) > 0:
                prediction.set_direction(original_image.direction)
            
            if np.linalg.norm(np.array(prediction.spacing) - np.array(original_image.spacing)) > 0:
                prediction = ants.resample_image(prediction, 
                                                    resample_params = list(original_image.spacing), 
                                                    use_voxels = False, 
                                                    interp_type = 1)
                        
            # Take only foreground components with min voxels 
            prediction_binary = ants.get_mask(prediction, cleanup = 0)
            prediction_binary = ants.label_clusters(prediction_binary, self.min_component_size)
            prediction_binary = ants.get_mask(prediction_binary, cleanup = 0)
            prediction_binary = prediction_binary.numpy()
            prediction = np.multiply(prediction_binary, prediction.numpy())
            
            prediction_dims = prediction.shape
            orignal_dims = original_image.numpy().shape
            prediction_final_dims = [np.max([prediction_dims[i], orignal_dims[i]]) for i in range(3)]
            
            prediction_final = np.zeros(tuple(prediction_final_dims))
            prediction_final[0:prediction.shape[0], 
                                0:prediction.shape[1], 
                                0:prediction.shape[2], ...] = prediction
            
            prediction_final = prediction_final[0:orignal_dims[0], 
                                                0:orignal_dims[1],
                                                0:orignal_dims[2], ...]
            
            prediction_final = original_mask.new_image_like(data = prediction_final)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction_final, 
                                os.path.join(self.prediction_dir, prediction_filename))
            
            # Get dice and hausdorff distance for final prediction
            row_dict = dict.fromkeys(list(self.results_df.columns))
            row_dict['id'] = patient['id']
            for key in self.final_classes.keys():
                class_labels = self.final_classes[key]
                pred = prediction_final.numpy()
                mask = original_mask.numpy()
                
                pred_temp = np.zeros(pred.shape)
                mask_temp = np.zeros(mask.shape)
                
                for label in class_labels:
                    pred_label = (pred == label).astype(np.uint8)
                    mask_label = (mask == label).astype(np.uint8)
                    
                    pred_temp += pred_label
                    mask_temp += mask_label
                    
                pred_temp = prediction_final.new_image_like(pred_temp)
                mask_temp = original_mask.new_image_like(mask_temp)
                
                pred_temp_filename = os.path.join(self.prediction_dir, 'pred_temp.nii.gz')
                ants.image_write(pred_temp, pred_temp_filename)
                
                mask_temp_filename = os.path.join(self.prediction_dir, 'mask_temp.nii.gz')
                ants.image_write(mask_temp, mask_temp_filename)
                
                row_dict['{}_dice'.format(key)] = self.dice_sitk(pred_temp_filename, mask_temp_filename)
                row_dict['{}_haus95'.format(key)] = self.hausdorff(pred_temp_filename, mask_temp_filename, '95')
                row_dict['{}_avg_surf'.format(key)] = self.surface_hausdorff(pred_temp_filename, mask_temp_filename, 'mean')
                
            print("Writing to results_df_______________________")
            self.results_df = self.results_df.append(row_dict, ignore_index = True)
            
            gc.collect()
            cnt += 1
            
        # Delete temporary files and iterator to reduce memory consumption
        del iterator
        os.remove(pred_temp_filename)
        os.remove(mask_temp_filename)
        gc.collect()

    @timeit
    def patient_loss(self, model, ds, LossFunc):
        
        val_loss = list()           
        iterator = ds.as_numpy_iterator()
        pred_time = list()
        for element in iterator:            
            image = element[0]
            truth = element[1]
            dims = image[..., 0].shape
                        
            padding = list()
            cropping = list()
            for i in range(3):
                if dims[i] % self.patch_size[i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / self.patch_size[i]) * self.patch_size[i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
                    
            # Pad image for sliding window inference
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape
                        
            # Start sliding window prediction
            strides = [patch_dim // 1 for patch_dim in self.patch_size]
            prediction = np.zeros((*pad_dims, self.n_classes))
            for i in range(0, pad_dims[0] - self.patch_size[0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.patch_size[1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.patch_size[2] + 1, strides[2]):
                        # Get image patch
                        patch = image[i:(i + self.patch_size[0]), 
                                        j:(j + self.patch_size[1]), 
                                        k:(k + self.patch_size[2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        prediction[i:(i + self.patch_size[0]), 
                                    j:(j + self.patch_size[1]), 
                                    k:(k + self.patch_size[2]), ...] = model.predict(patch)

            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]
            prediction = prediction.reshape((1, *prediction.shape)).astype('float32')            
            truth = truth.reshape((1, *truth.shape)).astype('float32')
            
            if self.dice:
                
                    val_loss = LossFunc.dice(truth, prediction)
            else:
                    val_loss = LossFunc.gdl(truth, prediction)
            # file1 = open("myfileTestLoss.txt", "a")
            # file1.writelines(str(epoch) + "\t"+ str(val_loss)+ "\t"+ str(self.loss) + "\t"+ str(self.model.optimizer.lr) )
            
            # file1.writelines('n')
            # file1.close()  
            gc.collect()
                    
        del iterator
        gc.collect()
        return val_loss

    def get_gaussian(self, sigma_scale = 0.125):
        tmp = np.zeros(self.patch_size)
        center_coords = [i // 2 for i in self.patch_size]
        sigmas = [i * sigma_scale for i in self.patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])
        
        gaussian_importance_map = gaussian_importance_map.reshape((*gaussian_importance_map.shape, 1))
        gaussian_importance_map = np.repeat(gaussian_importance_map, self.n_classes, axis = -1)

        return gaussian_importance_map

    def performanceMetrics(self):

        # row_dict = dict.fromkeys(list(self.results_df.columns))
        # for key in self.final_classes.keys():
        #     yDice = row_dict['{}_dice'.format(key)] 
        #     yHaus95 = row_dict['{}_haus95'.format(key)] 
        yDice = self.results_df['Liver_dice']
        yHaus95 = self.results_df['Liver_haus95']

        headers = [self.model_name, "Mean", "Median", "Min", "Max", "25th Percentile", "50th Percentile", "75th Percentile", "Std"]
    
 
        diceList = ["Dice Similarity Coefficient (" + self.model_name + ")", 
                                                    np.mean(yDice), 
                                                    np.median(yDice), 
                                                    min(yDice), 
                                                    max(yDice),
                                                    np.percentile(yDice, 25),
                                                    np.percentile(yDice, 50),
                                                    np.percentile(yDice, 75),
                                                    np.std(yDice)]
        hdList = ["95th--percentile Hausdorff Distance (" + self.model_name + ")", 
                                                    np.mean(yHaus95), 
                                                    np.median(yHaus95),  
                                                    min(yHaus95), 
                                                    max(yHaus95), 
                                                    np.percentile(yHaus95, 25),
                                                    np.percentile(yHaus95, 50),
                                                    np.percentile(yHaus95, 75),
                                                    np.std(yHaus95)]

        data = pd.DataFrame([ diceList, hdList], columns=headers)
        filename = self.model_name + '_'+ '{}_dice'.format('Liver')  + 'Data.csv'
        
        data.to_csv(os.path.join(self.resultsPath,filename), mode='a', index=False, header=False )

        mean_row = {'id': 'Mean'}
        median_row = {'id': 'Median'}
        std_row = {'id': 'Std'}
        percentile25_row = {'id': '25th Percentile'}
        percentile50_row = {'id': '50th Percentile'}
        percentile75_row = {'id': '75th Percentile'}

        results_cols = self.results_df.columns
        for col in results_cols[1:]:
            mean_row[col] = np.mean(self.results_df[col])
            std_row[col] = np.std(self.results_df[col])
            median_row[col] = np.median(self.results_df[col])
            percentile25_row[col] = np.percentile(self.results_df[col], 25)
            percentile50_row[col] = np.percentile(self.results_df[col], 50)
            percentile75_row[col] = np.percentile(self.results_df[col], 75)


        self.results_df = self.results_df.append(mean_row, ignore_index = True)
        self.results_df = self.results_df.append(std_row, ignore_index = True)
        self.results_df = self.results_df.append(median_row, ignore_index = True)
        self.results_df = self.results_df.append(percentile25_row, ignore_index = True)
        self.results_df = self.results_df.append(percentile50_row, ignore_index = True)
        self.results_df = self.results_df.append(percentile75_row, ignore_index = True)

        filename = self.model_name + '_'+ '{}_dice'.format('Liver')  + 'Data_previous.csv'
        os.path.join(self.resultsPath,filename)
        self.results_df.to_csv(os.path.join(self.resultsPath,filename), index = False)

        # Write results to csv file
        self.results_df.to_csv(self.resultsPath + 'results_csv', index = False)

    def plotMetrics(self, ):
        row_dict = dict.fromkeys(list(self.results_df.columns))
           
    
        Liver_dice = self.results_df['Liver_dice']
        Liver_haus95 = self.results_df['Liver_haus95']
        id = self.results_df['id']
        #Liver_dice = row_dict['{}_dice'.format(key)]
        #Liver_haus95 = row_dict['{}_haus95'.format(key)]
        id = self.results_df['id']

        yDice = [x for _, x in sorted(zip(id, Liver_dice))]
        yHaus95 = [x for _, x in sorted(zip(id, Liver_haus95))]
        x = range(len(id))

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Patient ID')
        ax1.set_ylabel('Dice coefficient', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(x, yDice, linestyle="-", marker="o", label='Dice coefficient', color=color)
        ax1.set_ylim(0, 1.05*max(yDice))


        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('HD Distance', color=color)
        ax2.plot(x, yHaus95, linestyle="-", marker="o",  label='95% HD Truth & Prediction', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.05*max(yHaus95))

        fig.tight_layout()

        ax2.set_title(self.model_name)

        plt.savefig(self.resultsPath + "/" + self.model_name + '.png', bbox_inches="tight")


    def plot_model_performance(self):
        fig, ax1 = plt.subplots(1)
        fig.suptitle('Model performance', size=20)
        ax1.plot(range(self.num_epochs), self.loss_end_of_epoch, label='Training loss' )
        
        ax1.set_ylabel('Loss', size = 14)
        ax1.set_xlabel('Epoch', size = 14)
        ax1.legend()
        plt.savefig(self.resultsPath + "/" + self.model_name + '.png', bbox_inches="tight")



