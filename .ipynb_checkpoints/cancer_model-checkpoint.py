            
#model

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

MODEL_PLOT_FILE = "model_plot.png"            
IMAGE_SIZE = 224

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
inputs = Input(input_shape)
xception = Xception(include_top=False, input_shape=input_shape)(inputs)
nas_net = NASNetMobile(include_top=False, input_shape=input_shape)(inputs)
outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(xception), GlobalAveragePooling2D()(nas_net)])
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)
model = Model(inputs, outputs)
model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
plot_model(model,
           to_file=MODEL_PLOT_FILE,
           show_shapes=True,
           show_layer_names=True)            
