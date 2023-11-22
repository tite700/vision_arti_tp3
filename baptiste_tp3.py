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


############################################################## CNN architecture personnelle
custom_cnn = Sequential()
custom_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
custom_cnn.add(MaxPooling2D(pool_size=(2, 2)))
custom_cnn.add(Conv2D(64, (3, 3), activation='relu'))
custom_cnn.add(MaxPooling2D(pool_size=(2, 2)))
custom_cnn.add(Flatten())
custom_cnn.add(Dense(128, activation='relu'))
custom_cnn.add(Dropout(0.5))
custom_cnn.add(Dense(8, activation='softmax'))

# Compilation du modèle
custom_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Affichage de la structure du modèle
custom_cnn.summary()

############################################################## CNN transfer learning VGG16

# Chargement du modèle VGG16 pré-entraîné sans les couches fully connected (include_top=False)
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Geler les couches pré-entraînées
for layer in vgg16_base.layers:
    layer.trainable = False

# Création du modèle en ajoutant des couches supplémentaires au-dessus des couches pré-entraînées
transfer_model = Sequential()
transfer_model.add(vgg16_base)

# Ajout de couches supplémentaires pour l'ajustement à votre tâche spécifique
transfer_model.add(Flatten())
transfer_model.add(Dense(256, activation='relu'))
transfer_model.add(Dropout(0.5))
transfer_model.add(Dense(8, activation='softmax'))

# Compilation du modèle
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Affichage de la structure du modèle
transfer_model.summary()

# Définition du nombre d'étapes par époque pour les ensembles d'entraînement, de validation et de test
train_steps_per_epoch = train_generator.n // train_generator.batch_size
val_steps_per_epoch = val_generator.n // val_generator.batch_size
test_steps_per_epoch = test_generator.n // test_generator.batch_size


######################################################################## Entraînement des modèles

# Entraînement du modèle CNN personnalisé
custom_cnn_history = custom_cnn.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=5,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch
)

# Entraînement du modèle de transfer learning avec VGG16
transfer_model_history = transfer_model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=5,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch
)

# Évaluation du modèle CNN personnalisé sur l'ensemble de test
custom_cnn_predictions = custom_cnn.predict(test_generator, steps=test_steps_per_epoch)
custom_cnn_pred_classes = np.argmax(custom_cnn_predictions, axis=1)
custom_cnn_true_classes = test_generator.classes[:test_steps_per_epoch * test_generator.batch_size]

# Évaluation du modèle de transfer learning avec VGG16 sur l'ensemble de test
transfer_model_predictions = transfer_model.predict(test_generator, steps=test_steps_per_epoch)
transfer_model_pred_classes = np.argmax(transfer_model_predictions, axis=1)
transfer_model_true_classes = test_generator.classes[:test_steps_per_epoch * test_generator.batch_size]

print("Nombre de classes dans l'ensemble d'entraînement :", train_generator.num_classes)
print("Nombre de classes dans l'ensemble de validation :", val_generator.num_classes)
print("Nombre de classes dans l'ensemble de test :", test_generator.num_classes)


# Calcul de l'accuracy global du classeur
custom_cnn_accuracy = accuracy_score(custom_cnn_true_classes, custom_cnn_pred_classes)
transfer_model_accuracy = accuracy_score(transfer_model_true_classes, transfer_model_pred_classes)

# Calcul du coefficient de kappa du classeur
custom_cnn_kappa = cohen_kappa_score(custom_cnn_true_classes, custom_cnn_pred_classes)
transfer_model_kappa = cohen_kappa_score(transfer_model_true_classes, transfer_model_pred_classes)

# Production du rapport
custom_cnn_report = classification_report(custom_cnn_true_classes, custom_cnn_pred_classes)
transfer_model_report = classification_report(transfer_model_true_classes, transfer_model_pred_classes)

# Affichage des résultats
print("Custom CNN - Accuracy:", custom_cnn_accuracy, "Kappa:", custom_cnn_kappa)
print("Transfer Learning (VGG16) - Accuracy:", transfer_model_accuracy, "Kappa:", transfer_model_kappa)
print("Custom CNN Report:\n", custom_cnn_report)
print("Transfer Learning (VGG16) Report:\n", transfer_model_report)

# Production des courbes d’apprentissage (entraînement et validation)
plt.figure(figsize=(12, 6))

# Courbe d'apprentissage du modèle CNN personnalisé
plt.subplot(1, 2, 1)
plt.plot(custom_cnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(custom_cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Custom CNN - Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Courbe d'apprentissage du modèle de transfer learning avec VGG16
plt.subplot(1, 2, 2)
plt.plot(transfer_model_history.history['accuracy'], label='Training Accuracy')
plt.plot(transfer_model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning (VGG16) - Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
