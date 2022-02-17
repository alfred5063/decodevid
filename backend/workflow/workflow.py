# Main Script

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import categorical_accuracy, PrecisionAtRecall, Recall
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from backend.loading.xray_dataload import *
from datetime import datetime
import os.path
import matplotlib.pyplot  as plt
from PIL import Image
from pathlib import Path
import imagesize
import PIL
from PIL import Image
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import cv2
from tensorflow.keras import backend as K
K.image_ordering_dim = "tf"

mainpath = r"D:\ALFRED - Workspace\Xray Images"
currentpath = r"D:\ALFRED - Workspace\Xray Images\Analysis - RGB - Test 2"
data_type = ['train_dataset', 'test_dataset']
labels = ['patients_covid_train', 'patients_covid_test']
batch_size = 25
img_width = 128
img_height = 128
epochs_range = 300
Dropout = 0.15
color_mode = 'rgb'
if color_mode == 'grayscale':
    img_dim = 1
elif color_mode == 'rgb':
    img_dim = 3
input_shape = (img_width, img_height, img_dim)
    
class_mode = 'binary'
rescale = 1./255              # normalize pixel values between 0-1
brightness_range = [0.1, 0.9] # specify the range in which to decrease/increase brightness
width_shift_range = 0.5       # shift the width of the image 50%
rotation_range = 90           # random rotation by 90 degrees
horizontal_flip = True        # 180 degree flip horizontally
vertical_flip = True          # 180 degree flip vertically
validation_split = 0.40
optimizer = Adam(lr = 0.0001) # RMSprop(lr = 0.001)
monitor = 'val_loss'
patience = 300

def getdate():
    try:
        print("Date and Timestamp")
        current_year = datetime.now().year
        current_timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_date = time.strftime("%Y%m%d")
        return current_year, current_timestamp, run_date
    except Exception as error:
        print("Error in Date and Timestamp", error)
        
'''        
def loadata_new(mainpath, run_date, data_type):
    try:
        list = []
        print("Check if current folder exists. If not, create a new one.")
        currentpath = mainpath + "\\RUN_" + run_date
            if not os.path.exists(currentpath):
                os.makedirs(currentpath)
                os.makedirs(currentpath + "\\train_dataset")
                os.makedirs(currentpath + "\\test_dataset")
                os.makedirs(currentpath + "\\validate_dataset")
                
        x_train = []
        for folder in os.listdir(currentpath + "\\train_dataset"):
            sub_path = train_path + "/" + folder
            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, (img_width, img_height))
                x_train.append(img_arr)
                
        x_test = []
        for folder in os.listdir(currentpath + "\\test_dataset"):
            sub_path = test_path + "/" + folder
            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, (img_width, img_height))
                x_test.append(img_arr)
        
        x_val = []
        for folder in os.listdir(currentpath + "\\validate_dataset"):
            sub_path = val_path + "/" + folder
            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, (img_width, img_height))
                x_val.append(img_arr)

        time.sleep(5)
    except Exception as error:
        print("Error in creating new run folder", error)
'''

# This part needs fixes
def loadata(mainpath, run_date, data_type):
    try:
        list = []
        print("Check if current folder exists. If not, create a new one.")
        currentpath = mainpath + "\\RUN_" + run_date
        if not os.path.exists(currentpath):
            os.makedirs(currentpath)
            os.makedirs(currentpath + "\\train_dataset")
            os.makedirs(currentpath + "\\train_dataset\\patients_covid_train")
            os.makedirs(currentpath + "\\train_dataset\\patients_covid_test")

            train_patients_covid = currentpath + "\\train_dataset\\patients_covid_train"
            list.append(train_patients_covid)
            train_patients_normal = currentpath + "\\train_dataset\\patients_covid_test"
            list.append(train_patients_normal)

            os.makedirs(currentpath + "\\test_dataset")
            os.makedirs(currentpath + "\\test_dataset\\patients_covid_train")
            os.makedirs(currentpath + "\\test_dataset\\patients_covid_test")

            test_patients_covid = currentpath + "\\test_dataset\\patients_covid_train"
            list.append(test_patients_covid)
            test_patients_normal = currentpath + "\\test_dataset\\patients_covid_test"
            list.append(test_patients_normal)

        time.sleep(5)
    except Exception as error:
        print("Error in creating new run folder", error)

    try:
        print("Data loading")
        labels = ['patients_covid_train', 'patients_covid_test']
        totalfile = sum_data(mainpath, labels)
        get_random_data(totalfile, currentpath, labels, data_type)
    except Exception as error:
        print("Error in data loading", error)

    return currentpath, list, labels, totalfile

def imd_eda(currentpath, labels, img_width, img_height, mainpath):
    
    # Rename files
    datasets = ['patients_covid_png', 'patients_lungopacity_png', 'patients_normal_png', 'patients_viralpneumonia_png', 'patients_covid_test', 'patients_covid_train']
    for dat in datasets:
        try:
            if dat == "patients_covid_png":
                name = "Covid"
            elif dat == "patients_lungopacity_png":
                name = "LungOpacity"
            elif dat == "patients_normal_png":
                name = "Normal"
            elif dat == "patients_viralpneumonia_png":
                name = "ViralPneumonia"
                
            files = os.listdir(mainpath + "\\Analysis - RGB - Test 2\\" + str(dat))
            i = 1
            for file in files:
                try:
                    os.rename(mainpath + "\\Analysis - RGB - Test 2\\" + str(dat) + "\\" + file, mainpath + "\\Analysis - RGB - Test 2\\" + str(dat) + "\\" + str(name) + "-" + str(i) + ".jpeg")
                    i = i+1
                except:
                    pass
                continue
        except:
            pass
        continue
    
    # Get the Image Resolutions
    folders = ['test_dataset', 'train_dataset']
    for a in folders:
        try:
            for b in labels:
                imgs = [img.name for img in Path(currentpath + "\\" + str(a) + "\\" + str(b)).iterdir() if img.suffix == ".png" or img.suffix == ".jpg" or img.suffix == ".jpeg"]
                img_meta = {}
                for f in imgs:
                    try:
                        img_meta[str(f)] = imagesize.get(currentpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f))
                        img = Image.open(currentpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f))
                        img = img.resize((img_width, img_height), PIL.Image.ANTIALIAS)
                        img.save(currentpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f))
                    except:
                        pass
                    continue
        except:
            pass
        continue
    
    # Convert it to Dataframe and compute aspect ratio
    #img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
    #img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
    #img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)
    
    # Visualize Image Resolutions
    
    #fig = plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(111)
    #points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color='blue', alpha=0.5, s=img_meta_df["Aspect Ratio"]*100, picker=True)
    #ax.set_title("Image Resolution")
    #ax.set_xlabel("Width", size=14)
    #ax.set_ylabel("Height", size=14)
    
    #img_meta_df.to_csv(currentpath + "\\" + "image_sizes.csv")


def prep(currentpath, data_type, batch_size, labels):

    train_datagen = ImageDataGenerator(
        rescale = rescale,
        brightness_range = brightness_range,
        width_shift_range = width_shift_range,
        rotation_range = rotation_range,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip,
        validation_split = validation_split
    )
    training_set = train_datagen.flow_from_directory(str(currentpath + "\\" + data_type[0]),
                                                     target_size = (img_width, img_height),
                                                     batch_size = batch_size,
                                                     color_mode = color_mode,
                                                     subset = 'training',
                                                     class_mode = class_mode)

    test_datagen = ImageDataGenerator(
        rescale = rescale,
        brightness_range = brightness_range,
        width_shift_range = width_shift_range,
        rotation_range = rotation_range,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip,
        validation_split = validation_split
    )
    test_set = test_datagen.flow_from_directory(str(currentpath + "\\" + data_type[0]),
                                                target_size = (img_width, img_height),
                                                batch_size = batch_size,
                                                color_mode = color_mode,
                                                subset = 'validation',
                                                class_mode = class_mode)

    steps_per_epoch = training_set.samples // batch_size
    val_steps = test_set.samples // batch_size
    return training_set, test_set, steps_per_epoch, val_steps
def modelit(currentpath, mainpath, training_set, test_set, steps_per_epoch, val_steps, epochs_range, data_type):

    # Referenced from: https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/

    #### Input Layer ####
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (img_width, img_height, img_dim)))
    cnn.add(tf.keras.layers.BatchNormalization())
    
    #### Convolutional Layer 1 ####
    cnn.add(tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #### Convolutional Layer 2 ####
    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #### Convolutional Layer 3 ####
    cnn.add(tf.keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #### Convolutional Layer 4 ####
    cnn.add(tf.keras.layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #### Convolutional Layer 5 ####
    cnn.add(tf.keras.layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #### Fully-Connected Layer 1 ####
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(1024, activation = 'relu'))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropout))
    cnn.add(tf.keras.layers.Dense(1, activation = 'sigmoid')) #Sigmoid is used because we want to predict probability of Covid-19 infected category
    
    
    #cnn = Sequential()
    #cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (img_width, img_height, img_dim)))
    #cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    #cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #cnn.add(tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    #cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    #cnn.add(tf.keras.layers.Dropout(Dropout))
    
    #cnn.add(tf.keras.layers.Conv2D(1, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    #cnn.add(tf.keras.layers.GlobalAveragePooling2D())
    #cnn.add(Activation('sigmoid'))
    
    #### Fully-Connected Layer 4 ####
    cnn.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics  = ['accuracy'])
    cnn.summary()
    
    # Saves Keras model after each epoch
    checkpointer = ModelCheckpoint(filepath = 'D:\\ALFRED - Workspace\\Analytics\\model.h5', verbose = True, save_best_only = True)
    
    # Reduce learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor = monitor, patience = patience, factor = 0.5, min_lr = 0.00001)
    
    # Early stopping to prevent overtraining and to ensure decreasing validation lose
    early_stop = EarlyStopping(monitor = monitor, patience = patience, restore_best_weights = True, mode = 'min')
    
    # Modelling
    history = cnn.fit(training_set,
                      epochs = epochs_range,
                      steps_per_epoch = steps_per_epoch,
                      validation_data = test_set,
                      validation_steps = val_steps,
                      callbacks=[early_stop, checkpointer, learning_rate_reduction],
                      verbose = True)
    if img_dim == 3:
        cnn.save("D:\\ALFRED - Workspace\\Analytics\\model.h5", include_optimizer = False)
    else:
        cnn.save("D:\\ALFRED - Workspace\\Analytics\\model.h5", include_optimizer = True)
    
    # Evaluating the result
    losses = pd.DataFrame.from_dict(history.history)
    losses.to_csv(mainpath + "\\" + "result_losses.csv")

    # Keras Accuracy
    keras_score, keras_acc = cnn.evaluate(training_set, verbose = 0)


    #plt.figure(figsize = (15, 15))
    #plt.subplot(2, 2, 1)
    #plt.plot(epochs_range, losses['accuracy'], label = 'Training Accuracy')
    #plt.plot(epochs_range, history.history['val_accuracy'], label = 'Validation Accuracy')
    #plt.legend(loc = 'lower right')
    #plt.title('Training and Validation Accuracy')
    #plt.savefig(currentpath + "\\Epoch_" + epochs_range + "_training_validationaccuracy.png")

    #plt.subplot(2, 2, 2)
    #plt.plot(epochs_range, history.history['loss'], label = 'Training Accuracy')
    #plt.plot(epochs_range, history.history['val_loss'], label = 'Validation Accuracy')
    #plt.legend(loc = 'upper right')
    #plt.title('Training and Validation Loss')
    #plt.savefig(currentpath + "\\Epoch_" + epochs_range + "_training_validationloss.png")

    return losses, keras_score, keras_acc
def predicting(mainpath, label, img_width, img_height, img_dim):
    final_result = []
    if label != 'None':
        img_path = str(mainpath + "\\" + label)
        os.chdir(str(mainpath + "\\" + label))
    else:
        img_path = str(mainpath)
        os.chdir(str(mainpath))
    if os.path.isfile("D:\\ALFRED - Workspace\\Analytics\\model.h5") == True:
        try:
            cnn = load_model("D:\\ALFRED - Workspace\\Analytics\\model.h5")
        except Exception as error:
            print("Model cannot be loaded with error message as: %s", error)
    else:
        print("Path to model does not exist.")
    
    for file in os.listdir():
        file
        try:
            if label != 'None':
                test_image = image.load_img(str(mainpath + "\\" + label + "\\" + file), target_size = (img_width, img_height))
            else:
                test_image = image.load_img(str(mainpath + "\\" + file), target_size = (img_width, img_height))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)

            # CNN Model
            try:
                result = cnn.predict(test_image)
                if float(result*100) < 0.50:
                    cat = 'Normal'
                elif float(result*100) > 0.50 and float(result*100) < 0.65:
                    cat = 'Infected - Mild'
                elif float(result*100) > 0.66 and float(result*100) < 0.81:
                    cat = 'Infected - Medium'
                else:
                    cat = 'Infected - Severe'
                    
                resultline = img_path + "\\" + str(file), str(file), 'Predictions: %', float(result*100), str(cat)
                final_result.append(resultline)
                print(str(file), 'Predictions: %', float(result*100), str(cat))
            except:
                pass
            continue
        except:
            pass
        continue
    
    final_resultdf = pd.DataFrame(final_result)
    #if label != 'None':
    #    final_resultdf.to_csv(str(mainpath + "\\" + label + "\\final_resultdf.csv"))
    #else:
    #    final_resultdf.to_csv(str(mainpath + "\\final_resultdf.csv"))
    return final_resultdf
def assessment(mainpath, patients):

    final_resultdf = []
    infected = []
    infected_df = []
    normal = []
    normal_df = []

    final_resultdf = pd.read_csv(mainpath + "\\" + patients + "\\final_resultdf.csv")
    final_resultdf = final_resultdf.rename(columns = {'0': 'image_path',
                                                      '1': 'image',
                                                      '2': 'prediction',
                                                      '3': 'percent', 
                                                      '4': 'status'}, inplace = False).drop(columns = {'Unnamed: 0'})
    infected = sum(final_resultdf['status']=='Infected')
    infected_df = final_resultdf[final_resultdf['status']=='Infected']
    infected_df = infected_df.reset_index()
    infected_df = infected_df.drop(columns = {'index'})
    
    normal = sum(final_resultdf['status']=='Normal')
    normal_df = final_resultdf[final_resultdf['status']=='Normal']
    normal_df = normal_df.reset_index()
    normal_df = normal_df.drop(columns = {'index'})

    return infected, infected_df, normal, normal_df, final_resultdf
def rgb_analysis(dir, coviddata):

    mean_val = []
    std_dev_val = []
    max_val = []
    min_val = []

    coviddata['image_bin'] = coviddata['image_path'].map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

    for i in range(len(coviddata['image'])):
        mean_val.append(coviddata['image_bin'][i].mean())
        std_dev_val.append(np.std(coviddata['image_bin'][i]))
        max_val.append(coviddata['image_bin'][i].max())
        min_val.append(coviddata['image_bin'][i].min())

    coviddata['mean'] = mean_val
    coviddata['stedev'] = std_dev_val
    coviddata['max'] = max_val
    coviddata['min'] = min_val
    subt_mean_samples = coviddata['mean'].mean() - coviddata['mean']
    coviddata['subt_mean'] = subt_mean_samples
    ax = sns.displot(data = coviddata, x = 'mean', kind="kde")
    plt.title('Images Colour Mean Value Distribution', fontsize = 16, weight = 'bold')
    plt.savefig(dir + "\\Images_Colour_Mean_Value_Distribution.png")

    ax = sns.displot(data = coviddata, x = 'mean', kind="kde", hue = 'status')
    plt.title('Images Colour Mean Value Distribution by Class', fontsize = 16, weight = 'bold')
    plt.savefig(dir + "\\Images_Colour_Mean_Value_Distribution_by_Class.png")

    ax = sns.displot(data = coviddata, x = 'max', kind="kde", hue = 'status');
    plt.title('Images Colour Max Value Distribution by Class', fontsize = 16, weight = 'bold')
    plt.savefig(dir + "\\Images_Colour_Max_Value_Distribution_by_Class.png")

    ax = sns.displot(data = coviddata, x = 'min', kind="kde", hue = 'status');
    plt.title('Images Colour Min Value Distribution by Class', fontsize = 16, weight = 'bold')
    plt.savefig(dir + "\\Images_Colour_Min_Value_Distribution_by_Class.png")

    return coviddata

# Re-model?
current_year, current_timestamp, run_date = getdate()
currentpath, list, labels, totalfile = loadata(mainpath, run_date, data_type)
imd_eda(currentpath, labels, img_width, img_height)
training_set, test_set, steps_per_epoch, val_steps = prep(currentpath, data_type, batch_size, labels)
losses, keras_score, keras_acc = modelit(currentpath, mainpath, training_set, test_set, steps_per_epoch, val_steps, epochs_range, data_type)
losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()
losses.plot()
#epochdf_final = pd.DataFrame()
#while e <= 100:
#    acc, val_acc, loss, val_loss, keras_score, epochdf = modelit(currentpath, mainpath, training_set, test_set, steps_per_epoch, val_steps, e)
#    epochdf_final = epochdf_final.append(epochdf, ignore_index = True)
#    e += 1
#epochdf_final = epochdf_final.reset_index()
#epochdf_final = epochdf_final.drop(columns = {'index'}).drop_duplicates(subset = None, keep = 'first', inplace = False)
#epochdf_final.to_csv(currentpath + "\\Epoch_" + str(e) + "_final_resultdf.csv")

# Run IT
final_result = []
types = [
    'patients_viralpneumonia',
        'patients_pneumonia',
        'patients_normal_Kaggle_Control',
        'patients_normal_Kaggle',
        'patients_lungopacity',
        'patients_covid_Kaggle',
        'patients_covid_AlforCOVID',
        'patients_bacterialpneumonia'
         ]
for t in range(len(types)):
    t
    try:
        final_result = predicting(mainpath, str(types[t]), img_width, img_height, img_dim)
        if final_result.empty != True:
            final_resultdf = pd.DataFrame(final_result)
            final_resultdf.to_csv(mainpath + "\\" + str(types[t]) + "_prediction_result.csv")
        else:
            print(str(types[t]) + " failed 1")
            pass
    except Exception as e:
        print(str(types[t]) + " failed 2" + str(e))
        pass
    continue

# Re-assessments
current_year, current_timestamp, run_date = getdate()
infected, infected_df, normal, normal_df, final_resultdf = assessment(mainpath, 'patients_covid')
infected_dir, normal_dir, analyzed_dir = gettransf_images(mainpath, currentpath, infected_df, normal_df, 'patients_covid', run_date)
coviddata = rgb_analysis(analyzed_dir, final_resultdf)
final_result = predicting(infected_dir, 'None')# -*- coding: utf-8 -*-

