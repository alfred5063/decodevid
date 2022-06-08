# Main Script
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, time, psutil, gc
import os.path
import imagesize
import PIL
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import categorical_accuracy, PrecisionAtRecall, Recall
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.python.client import device_lib
from backend.loading.xray_dataload import *
from datetime import datetime
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from numpy import expand_dims
from matplotlib import pyplot
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K.image_ordering_dim = "tf"


currentpath = r"D:\ALFRED - Workspace\Xray Images\Analysis - RGB - Test 2\dataset_1\Set 1 - Copy\subset_1"
trainpath = str(currentpath) + "\\train_dataset"
testpath = str(currentpath) + "\\test_dataset"
#patient_covid_severe = r"D:\ALFRED - Workspace\Xray Images\From AIforCOVID\12 Nov 2021\archive\patient_covid_severe.xlsx"
#patient_covid_mild = r"D:\ALFRED - Workspace\Xray Images\From AIforCOVID\12 Nov 2021\archive\patient_covid_mild.xlsx"
#patient_covid_lieve = r"D:\ALFRED - Workspace\Xray Images\From AIforCOVID\12 Nov 2021\archive\patient_covid_lieve.xlsx"
#destination = r"D:\ALFRED - Workspace\Xray Images\Analysis - RGB - Test 2\dataset\train_dataset\patient_covid_severe"
#import openpyxl
#path = r'D:\ALFRED - Workspace\Xray Images\Analysis - RGB - Test 2'
#file = pd.read_excel(str(patient_covid_severe))
#for f in file['Picture']:
#    print(str(f))
#    try:
#        shutil.copy("D:\\ALFRED - Workspace\\Xray Images\\patients_covid_AlforCOVID\\" + str(f), destination)
#    except:
#        pass
#    continue

data_type = ['train_dataset', 'test_dataset']
classes = ["patient_covid_severe", "patient_covid_mild", "patient_covid_lieve", "patient_viralpneumonia", "patient_normal"]
labels = classes
balancing = ''

diag_code_dict = {
    'Covid_Severe': 0,
    'Covid_Mild': 1,
    'Covid_Lieve': 2,
    'ViralPneumonia': 3,
    'Normal': 4
    }
diag_title_dict = {
    'Covid_Severe': 'patient_covid_severe',
    'Covid_Mild': 'patient_covid_mild',
    'Covid_Lieve': 'patient_covid_lieve',
    'ViralPneumonia': 'patient_viralpneumonia',
    'Normal': 'patient_normal'
    }

activate = 'softmax'
if activate == 'sigmoid':
    class_len = 1
    loss = 'binary_crossentropy'
    class_mode = 'binary'
else:
    class_len = len(classes)
    loss = 'categorical_crossentropy'
    class_mode = 'categorical'
    
color_mode = 'rgb'
if color_mode == 'grayscale':
    img_dim = 1
elif color_mode == 'rgb':
    img_dim = 3

model_history = []
samples = ''
img_width = 128
img_height = 128
n_folds = 5
epochs_range = 250
Dropt = 0.05
rescale = 1./255              # normalize pixel values between 0-1
brightness_range = [0.1, 0.9] # specify the range in which to decrease/increase brightness
width_shift_range = 0.5       # shift the width of the image 50%
rotation_range = 90           # random rotation by 90 degrees
horizontal_flip = True        # 180 degree flip horizontally
vertical_flip = True          # 180 degree flip vertically
validation_split = 0.99
fill_mode = 'nearest'
learn_rate = 0.01
optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
patience = 10
input_shape = (img_width, img_height, img_dim)

# To define function to find batch size for training the model
# use this function to find out the batch size

def FindBatchSize(model):

    BatchFound = 16
    try:
        total_params= int(model.count_params())
        GCPU = "CPU"
        try:
            if K.tensorflow_backend._get_available_gpus() == []:
                GCPU = "CPU"
            else:
                GCPU = "GPU"
        except:
            def get_available_gpus():
                local_device_protos= device_lib.list_local_devices()
                return [x.name for x in local_device_protos if x.device_type == 'GPU']
            if "gpu" not in str(get_available_gpus()).lower():
                GCPU = "CPU"
            else:
                GCPU = "GPU"
        if (GCPU == "GPU") and (os.cpu_count() > 15) and (total_params < 100000):
            BatchFound = 64
        if (os.cpu_count() < 16) and (total_params <500000):
            BatchFound = 64  
        if (GCPU == "GPU") and (os.cpu_count() > 15) and (total_params < 200000) and (total_params >= 100000):
            BatchFound = 32
        if (GCPU == "GPU") and (os.cpu_count() > 15) and (total_params >= 200000) and (total_params < 100000):
            BatchFound = 16
        if (GCPU == "GPU") and (os.cpu_count() > 15) and (total_params >= 100000):
            BatchFound = 8
        if (os.cpu_count() < 16) and (total_params > 500000):
            BatchFound = 8
        if total_params > 100000000:
            BatchFound = 1
    except:
        pass
    try:
        memoryused = psutil.virtual_memory()
        memoryused = float(str(memoryused).replace(" ", "").split("percent=")[1].split(",")[0])
        if memoryused > 75.0:
            BatchFound = 8
        if memoryused > 85.0:
            BatchFound = 4
        if memoryused > 90.0:
            BatchFound = 2
        if total_params > 100000000:
            BatchFound = 1
        print("Batch Size:  "+ str(BatchFound))
        gc.collect()
    except:
        pass
    memoryused = [];    total_params = [];    GCPU = "";
    del memoryused, total_params, GCPU;    gc.collect()
    return BatchFound
    
def getdate():
    try:
        print("Date and Timestamp")
        current_year = datetime.now().year
        current_timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_date = time.strftime("%Y%m%d")
        return current_year, current_timestamp, run_date
    except Exception as error:
        print("Error in Date and Timestamp", error)

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

def imgprocess(currentpath, data_type, classes, img_width, img_height, mainpath):
    for dat in classes:
        dat
        try:
            if dat == "patient_covid_severe":
                name = "Covid_Severe"
            elif dat == "patient_covid_mild":
                name = "Covid_Mild"
            elif dat == "patient_covid_lieve":
                name = "Covid_Lieve"
            elif dat == "patient_viralpneumonia":
                name = "ViralPneumonia"
            elif dat == "patient_normal":
                name = "Normal"
            elif dat == "patient_covid":
                name = "Covid"
            for clas in data_type:
                files = os.listdir(mainpath + "\\" + str(clas) + "\\" + str(dat))
                print(files)
                for file in files:
                    file
                    print(file)
                    try:
                        img = load_img(mainpath + "\\" + str(clas) + "\\" + str(dat) + "\\" + str(file))
                        data = img_to_array(img)
                        sample = expand_dims(data, 0)
                        datagen = ImageDataGenerator(rotation_range = 50)
                        it = datagen.flow(sample, batch_size = 1)
                        i = 1
                        for i in range(10):
                            i
                            try:
                                #pyplot.subplot(330 + 1 + i)
                                batch = it.next()
                                image = batch[0].astype('uint8')
                                cv2.imwrite(mainpath + "\\" + str(clas) + "\\" + str(dat) + "\\" + str(i) + "_" + str(file), image)
                            except:
                                pass
                            continue
                    except:
                        pass
                    continue
            for clas in data_type:
                try:
                    files = os.listdir(mainpath + "\\" + str(clas) + "\\" + str(dat))
                    i = 1
                    for file in files:
                        try:
                            os.rename(mainpath + "\\" + str(clas) + "\\" + str(dat) + "\\" + file, mainpath + "\\" + str(clas) + "\\" + str(dat) + "\\" + str(name) + "-" + str(i) + ".jpeg")
                            i = i+1
                        except:
                            pass
                        continue
                except:
                    pass
                continue
        except:
            pass
        continue
    
    # Get the Image Resolutions
    for a in data_type:
        try:
            for b in classes:
                print(b)
                imgs = [img.name for img in Path(mainpath + "\\" + str(a) + "\\" + str(b)).iterdir() if img.suffix == ".png" or img.suffix == ".jpg" or img.suffix == ".jpeg"]
                img_meta = {}
                for f in imgs:
                    print("--- " + f)
                    try:
                        # Rescale and standardize size
                        img_meta[str(f)] = imagesize.get(mainpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f))
                        img = Image.open(mainpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f))
                        img = img.resize((img_width, img_height), PIL.Image.ANTIALIAS)
                        img.save(mainpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f))
                        
                        # Sharpening image
                        img = cv2.imread(mainpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f), flags = cv2.IMREAD_COLOR)
                        kernel = np.array([[0, -1, 0],
                                           [-1, 5,-1],
                                           [0, -1, 0]])
                        image_sharp = cv2.filter2D(src = img, ddepth = -1, kernel = kernel)
                        cv2.imwrite(mainpath + "\\" + str(a) + "\\" + str(b) + "\\" + str(f), img)
                    except:
                        pass
                    continue
        except:
            pass
        continue
    
    # Convert it to Dataframe and compute aspect ratio
    img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis = 'columns', inplace = False)
    img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index = img_meta_df.index)
    img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)
    
    # Visualize Image Resolutions
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111)
    points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color='blue', alpha = 0.5, s = img_meta_df["Aspect Ratio"]*100, picker=True)
    ax.set_title("Image Resolution")
    ax.set_xlabel("Width", size = 14)
    ax.set_ylabel("Height", size = 14)

    img_meta_df.to_csv(currentpath + "\\" + "image_sizes.csv")


def image_bal(trainpath, diag_code_dict, diag_title_dict, samples):
    try:
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(trainpath, '*','*.jpeg'))}
        covidData = pd.DataFrame.from_dict(imageid_path_dict, orient = 'index').reset_index()
        covidData.columns = ['image_id','path']
        classes = covidData.image_id.str.split('-').str[0]
        covidData['diag'] = classes
        covidData['target'] = covidData['diag'].map(diag_code_dict.get)
        covidData['Class'] = covidData['diag'].map(diag_title_dict.get)
        samples, features = covidData.shape
        duplicated = covidData.duplicated().sum()
        null_values = covidData.isnull().sum().sum()
        print('Basic EDA')
        print('Number of samples: %d'%(samples))
        print('Number of duplicated values: %d'%(duplicated))
        print('Number of Null samples: %d' % (null_values))
    
        # Samples per class
        plt.figure(figsize = (20,8))
        sns.set(style = "ticks", font_scale = 1)
        ax = sns.countplot(data = covidData, x = 'Class', order = covidData['Class'].value_counts().index, palette = "flare")
        sns.despine(top = True, right = True, left = True, bottom = False)
        plt.xticks(rotation = 0, fontsize = 12)
        ax.set_xlabel('Sample Type - Diagnosis', fontsize = 14, weight = 'bold')
        ax.set(yticklabels = [])
        ax.axes.get_yaxis().set_visible(False) 
        plt.title('Number of Samples per Class', fontsize = 16, weight = 'bold')
        
        # Plot numbers
        for p in ax.patches:
            ax.annotate("%.1f%%" % (100*float(p.get_height()/samples)), (p.get_x() + p.get_width() / 2., abs(p.get_height())),
            ha = 'center', va = 'bottom', color = 'black', xytext = (0, 10),rotation = 'horizontal',
            textcoords = 'offset points')
        maxval = max(covidData['Class'].value_counts())
        minval = min(covidData['Class'].value_counts())
        bal = maxval - minval
        if bal == 0:
            balancing = 'balance'
            metrics = 'accuracy'
            monitor = 'val_loss'
        else:
            balancing = 'imbalance'
            metrics = 'recall'
            monitor = 'val_recall'
        covidData['image'] = covidData['path'].map(lambda x: np.asarray(Image.open(x).resize((80, 80))))
        covidData = pd.DataFrame(covidData)
        covidData.to_csv(trainpath + "\\covidData.csv")
    except:
        covidData = pd.DataFrame()
        balancing = ''
        metrics = ''
        monitor = ''
    return covidData, balancing, metrics, monitor, samples
    
def img_profiling(covidData, trainpath, samples):

    plt.figure()
    pic_id = random.randrange(0, samples)
    picture = covidData['path'][pic_id]
    image = cv2.imread(picture)
    plt.imshow(image)
    plt.axis('off');
    plt.show()
    
    print('Shape of the image : {}'.format(image.shape))
    print('Image Hight {}'.format(image.shape[0]))
    print('Image Width {}'.format(image.shape[1]))
    print('Dimension of Image {}'.format(image.ndim))
    print('Image size {}'.format(image.size))
    print('Image Data Type {}'.format(image.dtype))
    print('Maximum RGB value in this image {}'.format(image.max()))
    print('Minimum RGB value in this image {}'.format(image.min()))
    
    image[0,0]
    plt.title('B channel',fontsize = 14,weight = 'bold')
    plt.imshow(image[ : , : , 0])
    plt.axis('off');
    plt.show()
    
    mean_val = []
    std_dev_val = []
    max_val = []
    min_val = []
    
    for i in range(0,samples):
        mean_val.append(covidData['image'][i].mean())
        std_dev_val.append(np.std(covidData['image'][i]))
        max_val.append(covidData['image'][i].max())
        min_val.append(covidData['image'][i].min())
    
    imageEDA = covidData.loc[:,['image', 'Class','path']]
    imageEDA['mean'] = mean_val
    imageEDA['stedev'] = std_dev_val
    imageEDA['max'] = max_val
    imageEDA['min'] = min_val
    
    subt_mean_samples = imageEDA['mean'].mean() - imageEDA['mean']
    imageEDA['subt_mean'] = subt_mean_samples
    
    imageEDApd = pd.DataFrame(imageEDA)
    imageEDApd.to_csv(trainpath + "\\imageEDA.csv")
    
    ax = sns.displot(data = imageEDA, x = 'mean', kind="kde");
    plt.title('Images Colour Mean Value Distribution', fontsize = 16,weight = 'bold');
    ax = sns.displot(data = imageEDA, x = imageEDA['mean'], kind="kde", hue = 'Class');
    plt.title('Images Colour Mean Value Distribution by Class', fontsize = 16,weight = 'bold');
    ax = sns.displot(data = imageEDA, x = 'max', kind="kde", hue = 'Class');
    plt.title('Images Colour Max Value Distribution by Class', fontsize = 16,weight = 'bold');
    ax = sns.displot(data = imageEDA, x = 'min', kind="kde", hue = 'Class');
    plt.title('Images Colour Min Value Distribution by Class', fontsize = 16,weight = 'bold');
    
    plt.figure(figsize = (20,8))
    sns.set(style = "ticks", font_scale = 1)
    ax = sns.scatterplot(data = imageEDA, x = 'mean', y = 'stedev', hue = 'Class', alpha = 0.8);
    sns.despine(top = True, right = True, left = False, bottom = False)
    plt.xticks(rotation = 0,fontsize = 12)
    ax.set_xlabel('Image Channel Colour Mean',fontsize = 14,weight = 'bold')
    ax.set_ylabel('Image Channel Colour Standard Deviation',fontsize = 14, weight = 'bold')
    plt.title('Mean and Standard Deviation of Image Samples', fontsize = 16, weight = 'bold');
    
    plt.figure(figsize=(20,8));
    g = sns.FacetGrid(imageEDA, col="Class",height=5);
    g.map_dataframe(sns.scatterplot, x='mean', y='stedev');
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size = 16)
    g.fig.subplots_adjust(top=.7)
    g.fig.suptitle('Mean and Standard Deviation of Image Samples',fontsize=16, weight = 'bold')
    axes = g.axes.flatten()
    axes[0].set_ylabel('Standard Deviation');
    for ax in axes:
        ax.set_xlabel('Mean')
    g.fig.tight_layout()


def prep(trainpath, testpath, batch_size, color_mode, class_mode, img_width, img_height, rescale, brightness_range, width_shift_range, rotation_range, horizontal_flip, vertical_flip):

    train_datagen = ImageDataGenerator(
        rescale = rescale,
        brightness_range = brightness_range,
        width_shift_range = width_shift_range,
        rotation_range = rotation_range,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip,
        #validation_split = validation_split,
        fill_mode = fill_mode
    )
    training_set = train_datagen.flow_from_directory(str(trainpath),
                                                     target_size = (img_width, img_height),
                                                     classes = classes,
                                                     shuffle = True,
                                                     batch_size = batch_size,
                                                     color_mode = color_mode,
                                                     subset = 'training',
                                                     class_mode = class_mode)
    test_datagen = ImageDataGenerator(
        rescale = rescale,
        #brightness_range = brightness_range,
        #width_shift_range = width_shift_range,
        #rotation_range = rotation_range,
        #horizontal_flip = horizontal_flip,
        #vertical_flip = vertical_flip,
        validation_split = validation_split,
        fill_mode = fill_mode
    )
    test_set = test_datagen.flow_from_directory(str(testpath),
                                                target_size = (img_width, img_height),
                                                classes = classes,
                                                shuffle = True,
                                                batch_size = batch_size,
                                                color_mode = color_mode,
                                                subset = 'validation',
                                                class_mode = class_mode)
    steps_per_epoch = training_set.samples // batch_size
    val_steps = test_set.samples // batch_size
    return training_set, test_set, steps_per_epoch, val_steps

# VGG-16 Model (CNN Architecture)
def model_vgg16(img_width, img_height, img_dim, Dropt, activate, loss, optimizer, metrics, mname = None):
    
    IMAGE_SIZE = [img_width, img_height]  # we will keep the image size as (64,64). You can increase the size for better results. 
    
    # loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
    vgg = VGG16(input_shape = IMAGE_SIZE + [img_dim], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG
    
    # this will exclude the initial layers from training phase as there are already been trained.
    #for layer in vgg.layers:
    #    layer.trainable = False
    
    # Add top layers
    model = tf.keras.models.Sequential()
    model.add(vgg)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(Dropt))
    model.add(tf.keras.layers.Dense(256, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(Dropt))
    model.add(tf.keras.layers.Dense(class_len, activation = activate))
    len(model.trainable_weights)
    
    #x = Flatten()(vgg.output)
    #x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
    #x = Dense(class_len, activation = activate)(x)  # adding the output layer with softmax function as this is a multi label classification problem.
    #model = Model(inputs = vgg.input, outputs = x)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    model.summary()
    
    # Determine the batch size
    batch_size = FindBatchSize(model)
    return model, batch_size

# ResNet-18 Model (CNN Architecture)
def model_resnet18(img_width, img_height, img_dim, num_classes, optimizer, metrics, activation, loss, mname = None):
    model = ResNet18(num_classes)
    model.build(input_shape = (None, img_width, img_height, img_dim))
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    model.summary()
    return model

# Decodevid Model (CNN Architecture)
def model_decodevid(Dropt, img_width, img_height, img_dim, class_len, optimizer, metrics, loss, mname = None):

    # Referenced from: https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/

    #### Input Layer ####
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (6, 6), padding = 'same', activation = 'relu', input_shape = (img_width, img_height, img_dim)))
    cnn.add(tf.keras.layers.BatchNormalization())
      
    #### Convolutional Layer 1 ####
    cnn.add(tf.keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropt))
    
    #### Convolutional Layer 2 ####
    #cnn.add(tf.keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
    #cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
    #cnn.add(tf.keras.layers.BatchNormalization())
    #cnn.add(tf.keras.layers.Dropout(Dropt))
    
    #### Convolutional Layer 3 ####
    cnn.add(tf.keras.layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropt))
    
    #### Convolutional Layer 4 ####
    #cnn.add(tf.keras.layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu'))
    #cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
    #cnn.add(tf.keras.layers.BatchNormalization())
    #cnn.add(tf.keras.layers.Dropout(Dropt))
    
    #### Convolutional Layer 5 ####
    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropt))
    
    #### Convolutional Layer 5 ####
    cnn.add(tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropt))
    
    #### Fully-Connected Layer 1 ####
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(class_len * 32, activation = 'relu'))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(Dropt))
    cnn.add(tf.keras.layers.Dense(class_len, activation = activate)) #Sigmoid is used because we want to predict probability of Covid-19 infected category
    
    #### Fully-Connected Layer 4 ####
    cnn.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    cnn.summary()
    
    # Determine the batch size
    batch_size = FindBatchSize(cnn)
    
    return cnn, batch_size


def crossvalidation(n_folds, model, currentpath, monitor, patience, training_set, epochs_range, steps_per_epoch, test_set, val_steps, mname, classes, vertical_flip):
    
    history = []
    for i in range(n_folds):
        print("Training on Fold: ", i+1)
        
        training_set, test_set, steps_per_epoch, val_steps = prep(
            trainpath, testpath, batch_size, color_mode, class_mode, img_width, img_height, rescale, brightness_range, width_shift_range, rotation_range, horizontal_flip, vertical_flip
            )
        
        history = model_training(
                learn_rate, model, currentpath, monitor, patience, training_set,
                epochs_range, steps_per_epoch, test_set, val_steps, classes, mname = mname, fold = i
                )
            
        model_history.append(history)

        print("=======" * 12, end = "\n\n\n")
        
    plt.title('Accuracies vs Epochs')
    plt.plot(model_history[0].history['accuracy'], label = 'Training Fold 1')
    plt.plot(model_history[1].history['accuracy'], label = 'Training Fold 2')
    plt.plot(model_history[2].history['accuracy'], label = 'Training Fold 3')
    plt.plot(model_history[3].history['accuracy'], label = 'Training Fold 4')
    plt.plot(model_history[4].history['accuracy'], label = 'Training Fold 5')
    plt.legend()
    plt.show()
    
    plt.title('Train Accuracy vs Val Accuracy')
    plt.plot(model_history[0].history['accuracy'], label = 'Train Accuracy Fold 1', color = 'black')
    plt.plot(model_history[0].history['val_accuracy'], label = 'Val Accuracy Fold 1', color = 'black', linestyle = "dashdot")
    plt.plot(model_history[1].history['accuracy'], label = 'Train Accuracy Fold 2', color = 'red', )
    plt.plot(model_history[1].history['val_accuracy'], label = 'Val Accuracy Fold 2', color = 'red', linestyle = "dashdot")
    plt.plot(model_history[2].history['accuracy'], label = 'Train Accuracy Fold 3', color = 'green', )
    plt.plot(model_history[2].history['val_accuracy'], label = 'Val Accuracy Fold 3', color = 'green', linestyle = "dashdot")
    plt.plot(model_history[3].history['accuracy'], label = 'Train Accuracy Fold 4', color = 'green', )
    plt.plot(model_history[3].history['val_accuracy'], label = 'Val Accuracy Fold 4', color = 'green', linestyle = "dashdot")
    plt.plot(model_history[4].history['accuracy'], label = 'Train Accuracy Fold 5', color = 'green', )
    plt.plot(model_history[4].history['val_accuracy'], label = 'Val Accuracy Fold 5', color = 'green', linestyle = "dashdot")
    plt.legend()
    plt.show()

    return model_history


# Train the model
def model_training(learn_rate, model, currentpath, monitor, patience, training_set, epochs_range, steps_per_epoch, test_set, val_steps, classes, mname = None, fold = None):
    
    # Saves Keras model after each epoch
    checkpointer = ModelCheckpoint(filepath = str(currentpath + "\\model.h5"), verbose = True, save_best_only = True)
    
    # Reduce learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor = monitor, patience = 1, factor = 0.5, min_lr = learn_rate)
    
    # Early stopping to prevent overtraining and to ensure decreasing validation lose
    early_stop = EarlyStopping(monitor = monitor, patience = patience, restore_best_weights = True, mode = 'min')
    
    # Modelling
    modelhistory = model.fit(training_set,
                      epochs = epochs_range,
                      steps_per_epoch = steps_per_epoch,
                      validation_data = test_set,
                      validation_steps = val_steps,
                      callbacks = [learning_rate_reduction, checkpointer, early_stop],
                      verbose = True)
    
    model.save(str(currentpath + "\\model_" + str(mname) + "_fold_" + str(fold) + ".h5"), include_optimizer = True)
    
    # Evaluating the result
    losses = pd.DataFrame.from_dict(modelhistory.history)
    losses.to_csv(currentpath + "\\model_" + str(mname) + "_fold_" + str(fold) + "_result_losses.csv")
    
    # Test model's accuracy - Testing Data
    model_accuracy(model, test_set, modelhistory, classes)
    
    # Keras Accuracy on Training Set
    trainkeras_score, trainkeras_acc = model.evaluate(training_set, verbose = 0)
    print("Keras Accuracy on Training Set:-")
    print("Score: %.2f%%" %(trainkeras_score*100))
    print("Accuracy: %.2f%%" %(trainkeras_acc*100))
    
    # Keras Accuracy on Testing Set
    testkeras_score, testkeras_acc = model.evaluate(test_set, verbose = 0)
    print("Keras Accuracy on Testing Set:-")
    print("Score: %.2f%%" %(testkeras_score*100))
    print("Accuracy: %.2f%%" %(testkeras_acc*100))
    
    return modelhistory

# Prediction
def predicting(mainpath, label, img_width, img_height, img_dim):
    final_result = []
    if label != 'None':
        img_path = str(mainpath + "\\" + label)
        os.chdir(str(mainpath + "\\" + label))
    else:
        img_path = str(mainpath)
        os.chdir(str(mainpath))
    if os.path.isfile(str(currentpath + "\\model - Fold 3 - 98acc.h5")) == True:
        try:
            cnn = load_model(str(currentpath + "\\model - Fold 3 - 98acc.h5"))
        except Exception as error:
            print("Model cannot be loaded with error message as: %s", error)
    else:
        print("Path to model does not exist.")
    
    for file in os.listdir():
        file
        try:
            if label != 'None':
                #test_image = cv2.imread(str(mainpath + "\\" + label + "\\" + file), flags = cv2.IMREAD_COLOR)
                #kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                #image_sharp = cv2.filter2D(src = test_image, ddepth = -1, kernel = kernel)
                #cv2.imwrite(str(mainpath + "\\" + label + "\\" + file), image_sharp)
                test_image = image.load_img(str(mainpath + "\\" + label + "\\" + file), target_size = (img_width, img_height))
            else:
                #test_image = cv2.imread(str(mainpath + "\\" + file), flags = cv2.IMREAD_COLOR)
                #kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                #image_sharp = cv2.filter2D(src = test_image, ddepth = -1, kernel = kernel)
                #cv2.imwrite(str(mainpath + "\\" + file), image_sharp)
                test_image = image.load_img(str(mainpath + "\\" + file), target_size = (img_width, img_height))
            
            # CNN Model
            try:
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                whichclass = np.vstack([test_image])
                result = cnn.predict(test_image)
                result = float(result[0][0])
                
                #if result < 50:
                #    cat = 'Normal'
                #elif result > 50 and result < 65:
                #    cat = 'Infected - Mild'
                #elif result > 66 and result < 81:
                #    cat = 'Infected - Medium'
                #else:
                #    cat = 'Infected - Severe'
                
                if result == 0:
                    cat = 'Infected - Mild'
                else:
                    cat = 'Infected - Severe'
                    
                resultline = img_path + "\\" + str(file), str(file), 'Predictions: Class ', str(result), str(cat)
                final_result.append(resultline)
                print(str(file), 'Predictions: Class ', str(result), str(cat))
            except:
                pass
            continue
        except:
            pass
        continue

    final_resultdf = pd.DataFrame(final_result)

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

# Modeling Accuracy
def model_accuracy(model, dataset, history, kelas):
    y_pred = model.predict(dataset)
    
    #Plot training and validation Loss
    fig, axarr = plt.subplots(1, 3, figsize = (15, 5), sharex = True)
    sns.set(style = "ticks", font_scale = 1)
    sns.despine(top = True, right = True, left = False, bottom = False)
    historyDF = pd.DataFrame.from_dict(history.history)
    ax = sns.lineplot(x = historyDF.index, y = history.history['accuracy'], ax = axarr[0], label = "Training");
    ax = sns.lineplot(x = historyDF.index, y = history.history['val_accuracy'], ax = axarr[0], label = "Validation");
    ax.set_ylabel('Accuracy')
    ax = sns.lineplot(x = historyDF.index, y = history.history['loss'], ax = axarr[1], label = "Training");
    ax = sns.lineplot(x = historyDF.index, y = history.history['val_loss'], ax = axarr[1], label = "Validation");
    ax.set_ylabel('Loss')
    try:
        ax = sns.lineplot(x = historyDF.index, y = history.history['learningrate'],ax = axarr[2]);
        ax.set_ylabel('Learning Rate')
    except:
        pass
    axarr[0].set_title("Training and Validation Set - Metric Recall")
    axarr[1].set_title("Training and Validation Set - Loss")
    axarr[2].set_title("Learning Rate during Training")
    
    for ax in axarr:
        ax.set_xlabel('Epochs')
        
    plt.suptitle('Training Performance Plots', fontsize = 16, weight = 'bold');
    fig.tight_layout(pad = 3.0)      
    plt.show()
    
    predictions = np.array(list(map(lambda x: np.argmax(x), y_pred)))
    y_true = dataset.classes
    CMatrix = pd.DataFrame(confusion_matrix(y_true, predictions), columns = classes, index = classes)
    len(y_true)
    plt.figure(figsize = (12, 6))
    ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'Blues')
    ax.set_xlabel('Predicted', fontsize = 14, weight = 'bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0);
    ax.set_ylabel('Actual', fontsize = 14, weight = 'bold') 
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0);
    ax.set_title('Confusion Matrix - Test Set', fontsize = 16, weight = 'bold', pad = 20);
    
    acc = accuracy_score(y_true, predictions)
    results_all = precision_recall_fscore_support(y_true, predictions, average = 'macro', zero_division = 1)
    results_class = precision_recall_fscore_support(y_true, predictions, average = None, zero_division = 1)
    metric_columns = ['Precision', 'Recall', 'F-Score', 'S']
    all_df = pd.concat([pd.DataFrame(list(results_class)).T, pd.DataFrame(list(results_all)).T])
    all_df.columns = metric_columns
    #all_df.index = ['Covid_Lieve','Covid_Severe','Covid_Mild','Covid_Lieve','Normal','ViralPneumonia','nan']
    
    def metrics_plot(df, metric):
        plt.figure(figsize = (22,10))
        ax = sns.barplot(data = df, x = df.index, y = metric, palette = "Blues_d")
        #Bar Labels
        for p in ax.patches:
            ax.annotate("%.1f%%" % (100*p.get_height()), (p.get_x() + p.get_width() / 2., abs(p.get_height())),
            ha = 'center', va = 'bottom', color = 'black', xytext = (-3, 5), rotation = 'horizontal', textcoords = 'offset points')
        sns.despine(top = True, right = True, left = True, bottom = False)
        ax.set_xlabel('Class', fontsize = 14, weight = 'bold')
        ax.set_ylabel(metric, fontsize = 14, weight = 'bold')
        ax.set(yticklabels = [])
        ax.axes.get_yaxis().set_visible(False) 
        plt.title(metric + ' Results per Class', fontsize = 16, weight = 'bold');
        
    metrics_plot(all_df, 'Precision')
    metrics_plot(all_df, 'Recall')
    metrics_plot(all_df, 'F-Score')
    print('**Overall Results**')
    print('Accuracy Result: %.2f%%' %(acc*100))
    print('Precision Result: %.2f%%' %(all_df.iloc[len(kelas),0]*100))
    print('Recall Result: %.2f%%' %(all_df.iloc[len(kelas),1]*100))
    print('F-Score Result: %.2f%%' %(all_df.iloc[len(kelas),2]*100))
    
    # Now, lets which category has much incorrect predictions
    label_frac_error = 1 - np.diag(CMatrix) / np.sum(CMatrix, axis = 1)
    plt.bar(np.arange(len(label_frac_error)), label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')

# Re-model?
current_year, current_timestamp, run_date = getdate()
currentpath, list, labels, totalfile = loadata(mainpath, run_date, data_type)
imd_eda(currentpath, labels, img_width, img_height)

# Image profiling
covidData, balancing, metrics, monitor, samples = image_bal(trainpath, diag_code_dict, diag_title_dict, samples)
if covidData.empty == False:
    img_profiling(covidData, currentpath, samples)
else:
    print("covidData is empty.")

# Modeling
#mname = "vgg16"
#model, batch_size = model_vgg16(img_width, img_height, img_dim, Dropt, activate, loss, optimizer, metrics, mname)
mname = "decodevid"
model, batch_size = model_decodevid(Dropt, img_width, img_height, img_dim, class_len, optimizer, metrics, loss, mname)

# Data preparation
#batch_size = 80
training_set, test_set, steps_per_epoch, val_steps = prep(trainpath, testpath, batch_size, color_mode, class_mode, img_width, img_height, rescale, brightness_range, width_shift_range, rotation_range, horizontal_flip, vertical_flip)

# Train the model with K-fold Cross Val
cval_model_history = crossvalidation(n_folds, model, currentpath, monitor, patience, training_set, epochs_range, steps_per_epoch, test_set, val_steps, mname, classes, vertical_flip)

# Train the model
model, trainkeras_score, trainkeras_acc, testkeras_score, testkeras_acc, losses, history = model_training(learn_rate, model, currentpath, monitor, patience, training_set, epochs_range, steps_per_epoch, test_set, val_steps, mname)

# Test model's accuracy - Testing Data
model_accuracy(model, test_set, history)

# Test model's accuracy - Training Data
model_accuracy(model, training_set, history)

# Run IT
final_result = []
final_resultdf = []
for t in range(len(classes)):
    t
    try:
        final_result = predicting(currentpath, str(classes[t]), img_width, img_height, img_dim)
        if final_result.empty != True:
            final_resultdf = pd.DataFrame(final_result)
            final_resultdf.to_csv(mainpath + "\\" + str(classes[t]) + "_prediction_result.csv")
        else:
            print(str(classes[t]) + " failed 1")
            pass
    except Exception as e:
        print(str(classes[t]) + " failed 2" + str(e))
        pass
    continue

# Re-assessments
current_year, current_timestamp, run_date = getdate()
infected, infected_df, normal, normal_df, final_resultdf = assessment(mainpath, 'patients_covid')
infected_dir, normal_dir, analyzed_dir = gettransf_images(mainpath, currentpath, infected_df, normal_df, 'patients_covid', run_date)
coviddata = rgb_analysis(analyzed_dir, final_resultdf)
final_result = predicting(infected_dir, 'None')# -*- coding: utf-8 -*-

