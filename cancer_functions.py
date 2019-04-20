import pandas as pd
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from livelossplot import PlotLossesKeras
from keras.callbacks import CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np 

SAMPLE_COUNT=85000
TRAINING_RATIO=0.9
IMAGE_SIZE = 96
BATCH_SIZE = 18
MODEL_PLOT_FILE = "model_plot.png"
EPOCHS = 10
VERBOSITY = 1
MODEL_FILE = "model.h5"
TRAINING_LOGS_FILE = "training_logs.csv"
TRAINING_PLOT_FILE = "training.png"
VALIDATION_PLOT_FILE = "validation.png"
ROC_PLOT_FILE = "roc.png"
TESTING_BATCH_SIZE = 5000
KAGGLE_SUBMISSION_FILE = "kaggle_submission.csv"

def datasetup(SAMPLE_COUNT, TRAINING_RATIO):    
    training_dir = 'static\\train\\'
    data_frame = pd.DataFrame({'path': glob(os.path.join(training_dir,'*.tif'))})
    data_frame['id'] = data_frame.path.map(lambda x: x.split('\\')[2].split('.')[0]) 
    labels = pd.read_csv('traindata\\train_labels.csv')
    data_frame = data_frame.merge(labels, on = 'id')
    negatives = data_frame[data_frame.label == 0].sample(SAMPLE_COUNT)
    positives = data_frame[data_frame.label == 1].sample(SAMPLE_COUNT)
    data_frame = pd.concat([negatives, positives]).reset_index()
    data_frame = data_frame[['path', 'id', 'label']]
    data_frame['image'] = data_frame['path'].map(imread)
    #small_data_frame=data_frame[0:10]
    #small_frame=data_frame[85000:85010]
    #smalldata_frame=pd.concat([small_data_frame, small_frame]).reset_index()
    #smalldata_frame = smalldata_frame[['path', 'id', 'label']]
    #smalldata_frame['image'] = smalldata_frame['path'].map(imread)
    training_path = '../training'
    validation_path = '../validation'
    for folder in [training_path, validation_path]:
        for subfolder in ['0', '1']:
            path = os.path.join(folder, subfolder)
            os.makedirs(path, exist_ok=True)
    training, validation = train_test_split(data_frame, train_size=TRAINING_RATIO, stratify=data_frame['label'])
    data_frame.set_index('id', inplace=True)
    for images_and_path in [(training, training_path), (validation, validation_path)]:
        images = images_and_path[0]
        path = images_and_path[1]
        for image in images['id'].values:
            file_name = image + '.tif'
            label = str(data_frame.loc[image,'label'])
            destination = os.path.join(path, label, file_name)
            if not os.path.exists(destination):
                source = os.path.join('static/train', file_name)
                shutil.copyfile(source, destination)

    return (training_path, validation_path)


training_path, validation_path=datasetup(SAMPLE_COUNT, TRAINING_RATIO)

  
training_data_generator = ImageDataGenerator(rescale=1./255,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            rotation_range=180,
                                            zoom_range=0.4, 
                                            width_shift_range=0.3,
                                            height_shift_range=0.3,
                                            shear_range=0.3,
                                            channel_shift_range=0.3)

def generation(IMAGE_SIZE, BATCH_SIZE, training_data_generator,training_path, validation_path):
    training_generator = training_data_generator.flow_from_directory(training_path,
                                                                    target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                    batch_size=BATCH_SIZE,
                                                                    class_mode='binary')
    validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,
                                                                                target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                                batch_size=BATCH_SIZE,
                                                                                class_mode='binary')
    testing_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,
                                                                            target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                            batch_size=BATCH_SIZE,
                                                                            class_mode='binary',
                                                                            shuffle=False)
    return (training_generator, validation_generator, testing_generator)   

training_generator, validation_generator, testing_generator=generation(IMAGE_SIZE, BATCH_SIZE, training_data_generator,training_path, validation_path)

def cancermodel(IMAGE_SIZE, MODEL_PLOT_FILE):     
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    xception = Xception(include_top=False, input_tensor = input_tensor)(input_tensor)
    nas_net = NASNetMobile(include_top=False, input_tensor = input_tensor)(input_tensor)
    outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(xception), GlobalAveragePooling2D()(nas_net)])
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
    model = Model(input_tensor, outputs)
    model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
    #plot_model(model,
    #        to_file=MODEL_PLOT_FILE,
    #        show_shapes=True,
    #        show_layer_names=True)     
    return model

model=cancermodel(IMAGE_SIZE, MODEL_PLOT_FILE)

def training(EPOCHS, VERBOSITY, MODEL_FILE, TRAINING_LOGS_FILE, training_generator, validation_generator, model):       
    history = model.fit_generator(training_generator,
                                steps_per_epoch=len(training_generator), 
                                validation_data=validation_generator,
                                validation_steps=len(validation_generator),
                                epochs=EPOCHS,
                                verbose=VERBOSITY,
                                callbacks=[PlotLossesKeras(),
                                            ModelCheckpoint(MODEL_FILE,
                                                            monitor='val_acc',
                                                            verbose=VERBOSITY,
                                                            save_best_only=True,
                                                            mode='max'),
                                            CSVLogger(TRAINING_LOGS_FILE,
                                                    append=False,
                                                    separator=';')])
    return history

history=training(EPOCHS, VERBOSITY, MODEL_FILE, TRAINING_LOGS_FILE, training_generator, validation_generator, model)

def trainingplots(TRAINING_PLOT_FILE, VALIDATION_PLOT_FILE, history):        
    epochs = [i for i in range(1, len(history.history['loss'])+1)]
    plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
    plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
    plt.legend(loc='best')
    plt.title('training')
    plt.xlabel('epoch')
    plt.savefig(TRAINING_PLOT_FILE, bbox_inches='tight')
    plt.close()
    plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
    plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
    plt.legend(loc='best')
    plt.title('validation')
    plt.xlabel('epoch')
    plt.savefig(VALIDATION_PLOT_FILE, bbox_inches='tight')
    plt.close()

trainingplots(TRAINING_PLOT_FILE, VALIDATION_PLOT_FILE, history)

def rocplotfile(VERBOSITY, ROC_PLOT_FILE, MODEL_FILE, model, testing_generator):        
    model.load_weights(MODEL_FILE)
    predictions = model.predict_generator(testing_generator, steps=len(testing_generator), verbose=VERBOSITY)
    false_positive_rate, true_positive_rate, threshold = roc_curve(testing_generator.classes, predictions)
    area_under_curve = auc(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
    plt.close()
    return threshold

threshold=rocplotfile(VERBOSITY, ROC_PLOT_FILE, MODEL_FILE, model, testing_generator)

def kaggletesting(TESTING_BATCH_SIZE, KAGGLE_SUBMISSION_FILE, model):        
    testing_files = glob(os.path.join('static\\train\\','*.tif'))
    submission = pd.DataFrame()
    for index in range(0, len(testing_files), TESTING_BATCH_SIZE):
        data_frame = pd.DataFrame({'path': testing_files[index:index+TESTING_BATCH_SIZE]})
        data_frame['id'] = data_frame.path.map(lambda x: x.split('\\')[2].split(".")[0])
        data_frame['image'] = data_frame['path'].map(imread)
        images = np.stack(data_frame.image, axis=0)
        predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]
        predictions = np.array(predicted_labels)
        data_frame['label'] = predictions
        submission = pd.concat([submission, data_frame[["id", "label"]]])
    submission.to_csv(KAGGLE_SUBMISSION_FILE, index=False, header=True)
    return KAGGLE_SUBMISSION_FILE

KAGGLE_SUBMISSION_FILE=kaggletesting(TESTING_BATCH_SIZE, KAGGLE_SUBMISSION_FILE, model)