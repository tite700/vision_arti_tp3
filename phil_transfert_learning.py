import tensorflow as tf
import numpy as np
import keras
import os

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import matplotlib.pyplot as plt

######################################################################################### Paramètres et pré-traitement des données

# Chemin du dossier principal contenant les classes
base_dir = './dataset'

# Chemin des dossiers pour les ensembles d'entraînement, de validation et de test
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# class
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

# Création d'un générateur d'images pour les ensembles d'entraînement avec augmentation de données
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Création d'un générateur d'images pour les ensembles de validation et de test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des images pour les ensembles d'entraînement, de validation et de test
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=16,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(100, 100),
    batch_size=16,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=16,
    class_mode='categorical'
)

# Visualize dataset
plt.figure(figsize=(10, 10))
for images, labels in train_generator.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = np.argmax(labels[i])
        plt.title(class_names[index])
        plt.axis('off')

plt.show()