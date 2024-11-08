######

# IE4483 Artificial Intelligence and Data Mining
# Mini Project 2 - Dogs/Cats Binary Classification using CNN

######


#### Import statements ####

import os
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt


#### Rename directories ####

# Training data
train_dir = 'cifar10/train'


# Validation data
val_dir = 'cifar10/val'


# Test data
test_dir = 'cifar10/test'



#### Check quantities ####

#print('total training images:', len(os.listdir(train_cat_dir)) + len(os.listdir(train_dog_dir)))
#print('total validation images:', len(os.listdir(val_cat_dir)) + len(os.listdir(val_dog_dir)))
print('total test images:', len(os.listdir(test_dir)))


#### Image Processing ####

# Create objects for training and validation data
train_datagen = ImageDataGenerator(
    #rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,                         # Train model using image augmentation
    shear_range=0.2,                                # including scaling, rotation, shear and flipping
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
val_datagen = ImageDataGenerator()    # Validation data should not be augmented

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Pass data into respective objects, labelling data as cat or dog
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(32, 32), batch_size=32, class_mode='categorical', classes=classes)     # Parameters (1):
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(32, 32), batch_size=32, class_mode='categorical', classes=classes)           # Target size: 16,16.
                                                                                                                         # Batch size: 128
#### Create CNN Model ####

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))     # Regularization: L2 (variable)
model.add(Dense(10, activation='softmax'))  # suggested final layer and activation function                                            # Activation function: sigmoid (variable)

model.summary()   # Display model summary

model.compile(loss='binary_crossentropy',                       # Parameters (3):
              optimizer=optimizers.Adam(learning_rate=1e-3),    # Loss function: Binary cross-entropy
              metrics=['acc'])                                  # Optimizer: Adam
                                                                # Learning rate: 1e-3

#### Callbacks ####

# Save the optimal model with highest validation accuracy
checkpoint = ModelCheckpoint("cnn_model.keras", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
# Stop training if validation accuracy does not improve for 20 epochs
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')   # Patience=20 (variable)
# Reduce learning rate by 5% after each epoch
#lr_schedule = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)  


#### Fit the Model ####

history = model.fit(train_generator,                            # Parameters (4):
                    steps_per_epoch=100,                        # Epochs: 100 (variable)
                    epochs=100,                                 # Steps per epoch: 100
                    validation_data=val_generator,              # Validation steps: 10
                    validation_steps=10, 
                    callbacks=[checkpoint,early_stopping]       # callbacks ON/OFF (variable)
)

#### Evaluate Performance ####

# Save model results into variables
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot training and validation accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# Plot training and validation loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
