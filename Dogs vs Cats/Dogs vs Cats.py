import os, cv2, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def prep_data(images_tups, image_width, image_height):
    #images =np.ndarray((len(images_tups), image_height, image_width, 3), dtype=np.uint8)
    images = []
    labels = []
    for image_tup in images_tups:
        image = cv2.resize(cv2.imread(image_tup[0], cv2.IMREAD_COLOR), (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        #images[images_tups.index(image_tup)] = image
        images.append(image)
        labels.append(image_tup[1])
    return images, labels

# load input data
train_dir = './input/train/'
test_dir = './input/test/'
train_dogs = [(train_dir+i, 1) for i in os.listdir(train_dir) if 'dog' in i]
train_cats = [(train_dir+i, 0) for i in os.listdir(train_dir) if 'cat' in i]
# pick the first 1000 images from each set
train = train_dogs[:100] + train_cats[:100]
# pick the first 10 images from test set
test = [(test_dir+i, -1) for i in os.listdir(test_dir)][:10]
random.shuffle(train)
#prepare data
image_width = 150
image_height = 150
x_train, y_train = prep_data(train, image_width, image_height)
x_test, y_test = prep_data(test, image_width, image_height)

#create model
base_model = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(image_height, image_width, 3),
    pooling='max'
    )
output = Dense(1, activation='sigmoid')(base_model.output)
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    loss='binary_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy']
    )
#train model
model.fit(
    np.array(x_train), np.array(y_train), 
    batch_size=10, 
    epochs=10,
    )
#predict
predictions = model.predict(np.array(x_test))
#save model
model.save('model_keras.h5')







