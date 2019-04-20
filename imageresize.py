def imgrsz(uploadpath, sz):
    from PIL import Image
    import os


    for image_file_name in os.listdir(uploadpath):          
        im = Image.open(uploadpath+image_file_name)
        new_width  = sz
        new_height = sz
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(uploadpath + image_file_name)
    
def loadingmodel():
    from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D
    from keras.applications.xception import Xception
    from keras.applications.nasnet import NASNetMobile
    from keras.models import Model
    from keras.optimizers import Adam
    IMAGE_SIZE = 96
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
    # model to pillow
    return model

def analysing(TESTING_BATCH_SIZE, KAGGLE_SUBMISSION_FILE, model, uploadpath):    
    from glob import glob
    import os
    import pandas as pd
    import numpy as np 
    from skimage.io import imread    
    testing_files = glob(os.path.join(uploadpath,'*.tif'))
    submission = pd.DataFrame()
    for index in range(0, len(testing_files), TESTING_BATCH_SIZE):
        data_frame = pd.DataFrame({'path': testing_files[index:index+TESTING_BATCH_SIZE]})
        data_frame['id'] = data_frame.path.map(lambda x: x.split('\\')[2].split(".")[0])
        data_frame['image'] = data_frame['path'].map(imread)
        images = np.stack(data_frame.image, axis=0)
        predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]
        predictions = np.array(predicted_labels)
        predictions_int = np.round(predictions)
        print(predictions_int,predictions)
        data_frame['label'] = predictions_int

        submission = pd.concat([submission, data_frame[["id", "label"]]])
    submission.to_csv(KAGGLE_SUBMISSION_FILE, index=False, header=True)
  
