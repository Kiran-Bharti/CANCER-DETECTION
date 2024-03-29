#data setup
import pandas as pd
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.io import imread


SAMPLE_COUNT=85000
TRAINING_RATIO=0.9


training_dir = 'static/train\\'
data_frame = pd.DataFrame({'path': glob(os.path.join(training_dir,'*.tif'))})

data_frame['id'] = data_frame.path.map(lambda x: x.split('\\')[1].split('.')[0]) 

labels = pd.read_csv('traindata\\train_labels.csv')

data_frame = data_frame.merge(labels, on = 'id')


negatives = data_frame[data_frame.label == 0].sample(SAMPLE_COUNT)
positives = data_frame[data_frame.label == 1].sample(SAMPLE_COUNT)
data_frame = pd.concat([negatives, positives]).reset_index()
data_frame = data_frame[['path', 'id', 'label']]
data_frame['image'] = data_frame['path'].map(imread)

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
            source = os.path.join('static/train ', file_name)
            shutil.copyfile(source, destination)
