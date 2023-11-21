import keras
import numpy as np
from keras.layers import Rescaling
from keras.applications import VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

'''
Simple transfert learning model
'''

# Définir le chemin vers votre dossier d'images
data_dir = "./data/natural_images"

# Définir la taille des images en entrée de VGG16 (224x224 pixels)
img_size = (224, 224)

# Define the input size for cnn
input_shape = (224, 224, 3)

# Définir le batch size
batch_size = 128

# Define the number of class
num_class = 8

# Training dataset
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='both',
    seed=123,
    batch_size=batch_size,
    image_size=img_size,
    label_mode='categorical'
)

# Test dataset
test_ds = val_ds.take(1)
val_ds = val_ds.skip(1)

class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

# Visualize dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = np.argmax(labels[i])
        plt.title(class_names[index])
        plt.axis('off')

plt.show()

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Sequential([
        Conv2D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        Conv2D(32, kernel_size=3),
        Conv2D(16, kernel_size=3),
        Flatten(),
        Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(train_ds, epochs=5, validation_data=val_ds)

predictions = model.predict(test_ds)

plt.figure(figsize=(10, 10))
for images, labels in test_ds:
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index_class = np.argmax(labels[i])
        index_prediction = np.argmax(predictions[i])
        plt.title(f"class={class_names[index_class]}, prediction={class_names[index_prediction]}")
        plt.axis('off')

plt.show()

# Calculate the accuracy on the test dataset
for images, labels in test_ds:
    score = 0
    for i in range(len(images)):
        index_class = np.argmax(labels[i])
        index_prediction = np.argmax(predictions[i])
        if index_class == index_prediction:
            score += 1

    print('accuracy = ', score/len(images))