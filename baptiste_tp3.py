import tensorflow as tf
import numpy as np
import keras
#use VGG16 model
from keras.applications.vgg16 import VGG16

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Définir le chemin vers votre dossier d'images
data_dir = "./data"

# Définir la taille des images en entrée de VGG16 (224x224 pixels)
img_size = (224, 224)

# Définir le batch size
batch_size = 32

# Prétraitement des données et augmentation des images
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Utilisez 20% des données pour la validation
)

# Charger les données d'entraînement
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Spécifiez qu'il s'agit du sous-ensemble d'entraînement
)

# Charger les données de validation
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Spécifiez qu'il s'agit du sous-ensemble de validation
)

# Charger l'architecture pré-entraînée VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

# Créer un modèle séquentiel et ajouter les couches nécessaires
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher un résumé du modèle
model.summary()

# Entraîner le modèle
model.fit(train_generator, epochs=10, validation_data=validation_generator)
