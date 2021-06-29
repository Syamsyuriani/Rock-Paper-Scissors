#Import the required modules
import tensorflow as tf
import zipfile
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from keras.callbacks import EarlyStopping

from google.colab import files
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Upload data
!wget --no-check-certificate \
  https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip \
  -O /content/rock_paper_scissors.zip

#Extract zip file
zip_lokal = '/content/rock_paper_scissors.zip'
zip_ref = zipfile.ZipFile(zip_lokal, 'r')
zip_ref.extractall('/content')
zip_ref.close()

base_dir = '/content/rockpaperscissors/rps-cv-images'
os.listdir('/content/rockpaperscissors/rps-cv-images')

# Image Augmentation
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    fill_mode = 'nearest',
                    validation_split=0.4)
 
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.4)

# Split dataset
train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training')
 
validation_generator = test_datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

# Model building
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

# Plot Model
plot_model(model, show_shapes=True, show_layer_names=True)

# Compile model with 'stochastic gradient descent' optimizer and loss function 'categorical_crossentropy' 
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# Define early stopping by paying attention to the value of validation loss
#(if validation loss starts to increase after 5 epochs then training will be stopped)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


#  Train models with model.fit
history = model.fit(train_generator,
                    steps_per_epoch=25,
                    epochs=15,
                    validation_data=validation_generator,
                    validation_steps=20,
                    verbose=1,
                    callbacks=[early_stop])

# Plot loss and accuracy graph
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1.5)
plt.show()

# Predict new data
uploaded = files.upload()
 
for fig in uploaded.keys():
  img = image.load_img(fig, target_size=(150,150))
  img_plot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  class_ = model.predict(images, batch_size=32)

  print(fig,'\n')
  if class_[0][0]==1:
    print('paper')
  elif class_[0][1]==1:
    print('rock')
  else:
    print('scissors')

