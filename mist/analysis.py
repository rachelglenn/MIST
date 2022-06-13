
import numpy as np
import tensorflow.keras.backend as K
import gc
import os
import ants
import pandas as pd
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
# and decreases it exponentially after that.
# def scheduler(epoch):
#   if epoch < 10:
#     return 0.001
#   else:
#     return 0.001 * tf.math.exp(0.1 * (10 - epoch))
# LearningRateCallback = LearningRateScheduler(scheduler)

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
# https://github.com/better-data-science/TensorFlow
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, best_model_name, patch_size, n_classes, LossFunc, test_df, test_ds, final_classes, dice=True):
        #super(EarlyStoppingAtMinLoss, self).__init__()
                
        self.val_ds = val_ds
        self.best_model_name = best_model_name
        self.patch_size = patch_size
        self.dice = dice
        self.n_classes = n_classes
        self.LossFunc = LossFunc
        self.loss = None
        self.best_val_loss = np.inf
        self.plateau_cnt = 1.0

        self.time_started = None
        self.time_finsihed = None
        self.test_df = test_df
        self.test_ds = test_ds
        # TODO remove
        self.savePath = '/rsrch1/ip/rglenn1/SegmentationResults/plots'
        self.final_classes = final_classes

        self._loss, self._acc, self._val_loss, self._val_acc = [], [], [], []
    
            # Initialize results dataframe
        metrics = ['dice', 'haus95', 'avg_surf']
        results_cols = ['id']
        for metric in metrics: 
            for key in final_classes.keys():
                results_cols.append('{}_{}'.format(key, metric))
                
        self.results_df = pd.DataFrame(columns = results_cols)


    def _plot_model_performance(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Model performance', size=20)
        ax1.plot(range(self.num_epochs), self._loss, label='Training loss' )
        ax1.plot(range(self.num_epochs), self._loss, label='Training loss' )
        ax1.set_ylabel('Loss', size = 14)
        ax1.set_xlabel('Epoch', size = 14)
        ax1.legend()

        ax2.plot(range(self.num_epochs), self._acc, label='Training accuracy' )
        ax2.plot(range(self.num_epochs), self._val_acc, label='Validation accuracy' )
        ax2.set_ylabel('Accuracy', size = 14)
        ax2.set_xlabel('Epoch', size = 14)
        ax2.legend()   

    
    def on_train_begin(self, logs=None):
        self.time_started = datetime.now()
        print(f'Training Started | {self.time_started}\n')

    def on_epoch_begin(self, epoch, logs=None):
        self.time_current_epoch = datetime.now()


    def on_train_end(self, logs=None):
        self.time_finished = datetime.now()
        train_duration = str(self.timefinished - self.time_started)
        print(f'\nTraining Finished | {self.time_finished} | Duration: {train_duration}')
        
        tl = f"Training loss:     {logs['loss']:.5f}"
        ta = f"Training accuracy: {logs['accuracy']:.5f}"
        vl = f"Validation loss:   {logs['val_loss']:.5f}"
        va = f"Validation loss:   {logs['val_accuracy']:.5f}"

        print('\n'.join([tl, vl, ta, va]))
        self._plot_model_performance()
        titlename = self.best_model_name
        self._plotMetrics(self, self.savePath, titlename)

        # Get final statistics
        self.performanceMetrics(titlename)
        
        # Write results to csv file
        self.results_df.to_csv(self.params['results_csv'], index = False)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        # Comput loss for validation patients
        self.loss = logs['loss']
        best_val_loss = self.best_val_loss
        val_loss = self.compute_val_loss(self.model, self.val_ds, self.patch_size, self.n_classes, epoch, self.LossFunc, dice = self.dice)
        
        if val_loss < best_val_loss:
            print('Val loss IMPROVED from {} to {}'.format(best_val_loss, val_loss))
            self.best_val_loss = val_loss

            self.model.save(self.best_model_name)
            self.plateau_cnt = 1


            
            # model = load_model(best_model_name, custom_objects = {'loss': self.loss.loss_wrapper(alpha)})
            self.val_inference(model, self.test_df, self.test_ds)
            
            split_cnt += 1
            
            del test_ds, model
            K.clear_session()
            gc.collect()
        else:
            print('Val loss of DID NOT improve from {}'.format(best_val_loss))
           
            self.plateau_cnt += 1
        if self.plateau_cnt % 10:
            self.model.optimizer.lr = self.model.optimizer.lr*0.9
            print('Decreasing learning rate to {}'.format( self.model.optimizer.lr))
        
        epoch_dur = (datetime.now() - self.time_current_epoch).total_seconds()
        tl = logs['loss']
        ta = logs['accuracy']
        vl = logs['val_loss']
        va = logs['val_accuracy']

        self._loss.append(tl); self._acc.append(ta); self._val_loss.append(vl); self._val_acc.append(va)

        train_metrics = f"  Training Loss: {tl:.5f},   Training Accuracy: {ta:.5f}"
        valid_metircs = f"Validation Loss: {vl:.5f}, Validation Accuracy: {va:.5f}"

        print(f"Epoch: {epoch:4} | Runtime: {epoch_dur:3.f}s | {train_metrics} | {valid_metircs}")

       
    def performanceMetrics(self, titlename):
        
        row_dict = dict.fromkeys(list(self.results_df.columns))
        for key in self.params['final_classes'].keys():
            yDice = row_dict['{}_dice'.format(key)] 
            yHaus95 = row_dict['{}_haus95'.format(key)] 
            headers = [titlename, "Mean", "Median", "Min", "Max", "25th Percentile", "50th Percentile", "75th Percentile", "Std"]
        

            diceList = ["Dice Similarity Coefficient (" + titlename + ")", 
                                                    np.mean(yDice), 
                                                    np.median(yDice), 
                                                     min(yDice), 
                                                     max(yDice),
                                                     np.percentile(yDice, 25),
                                                     np.percentile(yDice, 50),
                                                     np.percentile(yDice, 75),
                                                      np.std(yDice)]
            hdList = ["95th--percentile Hausdorff Distance (" + titlename + ")", 
                                                        np.mean(yHaus95), 
                                                        np.median(yHaus95),  
                                                        min(yHaus95), 
                                                        max(yHaus95), 
                                                        np.percentile(yHaus95, 25),
                                                        np.percentile(yHaus95, 50),
                                                        np.percentile(yHaus95, 75),
                                                        np.std(yHaus95)]

            data = pd.DataFrame([ diceList, hdList], columns=headers)
            filename = titlename + '_'+ '{}_dice'.format(key)  + 'Data.csv'
            
            data.to_csv(os.path.join(self.savePath,filename), mode='a', index=False, header=False )



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

        filename = titlename + '_'+ '{}_dice'.format(key)  + 'Data_previous.csv'
        os.path.join(self.savePath,filename)
        self.results_df.to_csv(os.path.join(self.savePath,filename), index = False)


class ValidateInference(tf.keras.metrics.Metric):

    def __init__(self, model, df, ds):
        #super(ValidateInference, self).__init__(name=name, **kwargs)
        #self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.df = df
        self.ds = ds
        self.model = model

    def val_inference(self, df, ds):
        model = self.model
        cnt = 0
        gaussian_map = self.get_gaussian()
                
        iterator = ds.as_numpy_iterator()
        for element in iterator:
            
            patient = df.iloc[cnt].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            
            original_mask = ants.image_read(patient['mask'])
            original_image = ants.image_read(image_list[0])
            original_dims = ants.image_header_info(image_list[0])['dimensions']
            
            if self.inferred_params['use_nz_mask']:
                nzmask = ants.get_mask(original_image, cleanup = 0)
                original_cropped = ants.crop_image(original_mask, nzmask)
            
            image = element[0]
            truth = element[1]
            dims = image[..., 0].shape
                        
            padding = list()
            cropping = list()
            for i in range(3):
                if dims[i] % self.inferred_params['patch_size'][i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / self.inferred_params['patch_size'][i]) * self.inferred_params['patch_size'][i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape
            
            strides = [patch_dim // 2 for patch_dim in self.inferred_params['patch_size']]
            prediction = np.zeros((*pad_dims, self.n_classes))
            for i in range(0, pad_dims[0] - self.inferred_params['patch_size'][0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.inferred_params['patch_size'][1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.inferred_params['patch_size'][2] + 1, strides[2]):
                        patch = image[i:(i + self.inferred_params['patch_size'][0]),
                                      j:(j + self.inferred_params['patch_size'][1]),
                                      k:(k + self.inferred_params['patch_size'][2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        pred_patch = model.predict(patch)
                        pred_patch *= gaussian_map
                        prediction[i:(i + self.inferred_params['patch_size'][0]),
                                   j:(j + self.inferred_params['patch_size'][1]),
                                   k:(k + self.inferred_params['patch_size'][2]), ...] = pred_patch
            
            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]

            prediction = np.argmax(prediction, axis = -1)
            
            # Make sure that labels are correct in prediction
            for j in range(self.n_classes):
                prediction[prediction == j] = self.params['labels'][j]
                
            prediction = prediction.astype('float32')
                
            if self.inferred_params['use_nz_mask']:
                prediction = original_cropped.new_image_like(data = prediction)
                prediction = ants.decrop_image(prediction, original_mask)
            else:
                # Put prediction back into original image space
                prediction = ants.from_numpy(prediction)
                
            prediction.set_spacing(self.inferred_params['target_spacing'])
                        
            if np.linalg.norm(np.array(prediction.direction) - np.eye(3)) > 0:
                prediction.set_direction(original_image.direction)
            
            if np.linalg.norm(np.array(prediction.spacing) - np.array(original_image.spacing)) > 0:
                prediction = ants.resample_image(prediction, 
                                                 resample_params = list(original_image.spacing), 
                                                 use_voxels = False, 
                                                 interp_type = 1)
                        
            # Take only foreground components with min voxels 
            prediction_binary = ants.get_mask(prediction, cleanup = 0)
            prediction_binary = ants.label_clusters(prediction_binary, self.inferred_params['min_component_size'])
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
                             os.path.join(self.params['prediction_dir'], prediction_filename))
            
            # Get dice and hausdorff distance for final prediction
            row_dict = dict.fromkeys(list(self.results_df.columns))
            row_dict['id'] = patient['id']
            for key in self.params['final_classes'].keys():
                class_labels = self.params['final_classes'][key]
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
                
                pred_temp_filename = os.path.join(self.params['prediction_dir'], 'pred_temp.nii.gz')
                ants.image_write(pred_temp, pred_temp_filename)
                
                mask_temp_filename = os.path.join(self.params['prediction_dir'], 'mask_temp.nii.gz')
                ants.image_write(mask_temp, mask_temp_filename)
                
                row_dict['{}_dice'.format(key)] = self.metrics.dice_sitk(pred_temp_filename, mask_temp_filename)
                row_dict['{}_haus95'.format(key)] = self.metrics.hausdorff(pred_temp_filename, mask_temp_filename, '95')
                row_dict['{}_avg_surf'.format(key)] = self.metrics.surface_hausdorff(pred_temp_filename, mask_temp_filename, 'mean')
                
            self.results_df = self.results_df.append(row_dict, ignore_index = True)
            
            gc.collect()
            cnt += 1
            
        # Delete temporary files and iterator to reduce memory consumption
        del iterator
        os.remove(pred_temp_filename)
        os.remove(mask_temp_filename)
        gc.collect()
    

    def compute_val_loss(self, model, ds, patch_size, n_classes, epoch, LossFunc,  dice=True,):
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
                if dims[i] % patch_size[i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / patch_size[i]) * patch_size[i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
                    
            # Pad image for sliding window inference
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape
                        
            # Start sliding window prediction
            strides = [patch_dim // 1 for patch_dim in patch_size]
            prediction = np.zeros((*pad_dims, n_classes))
            for i in range(0, pad_dims[0] - patch_size[0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - patch_size[1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - patch_size[2] + 1, strides[2]):
                        # Get image patch
                        patch = image[i:(i + patch_size[0]), 
                                      j:(j + patch_size[1]), 
                                      k:(k + patch_size[2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        prediction[i:(i + patch_size[0]), 
                                   j:(j + patch_size[1]), 
                                   k:(k + patch_size[2]), ...] = model.predict(patch)

            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]
            prediction = prediction.reshape((1, *prediction.shape)).astype('float32')            
            truth = truth.reshape((1, *truth.shape)).astype('float32')
            
            if dice:
                
                 val_loss = LossFunc.dice(truth, prediction)
            else:
                 val_loss = LossFunc.gdl(truth, prediction)
            file1 = open("myfileTestLoss.txt", "a")
            file1.writelines(str(epoch) + "\t"+ str(val_loss)+ "\t"+ str(self.loss) + "\t"+ str(self.model.optimizer.lr) )
           
            file1.writelines('n')
            file1.close()  
            gc.collect()
                    
        del iterator
        gc.collect()
        return np.mean(val_loss)


    def plotMetrics(self, Liver_dice, Liver_haus95, id,savePath, titlename):

        #Liver_dice = hcc_results['Liver_dice']
        #Liver_haus95 = hcc_results['Liver_haus95']
        #id = hcc_results['id']
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

        ax2.set_title(titlename)

        plt.savefig(savePath + "/" + titlename + '.png', bbox_inches="tight")