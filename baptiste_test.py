import tensorflow as tf
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import matplotlib.pyplot as plt

# Définir le chemin vers votre dossier d'images
data_dir = "./data"

# Définir la taille des images en entrée de VGG16 
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
    subset='training',
    shuffle=True
)

# Charger les données de validation
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
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
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Évaluation du modèle sur l'ensemble de test
test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

predictions = model.predict(test_generator)
y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)

# Calculer et afficher les indicateurs de performance
accuracy = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())

print(f"Accuracy: {accuracy}")
print(f"Cohen's Kappa: {kappa}")
print("Classification Report:")
print(report)

# Afficher les courbes d'apprentissage
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Visualiser les prédictions
plt.figure(figsize=(10, 10))
for images, labels in test_generator:
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {test_generator.index_to_word[labels[i].argmax()]}\nPredicted: {test_generator.index_to_word[np.argmax(model.predict(np.expand_dims(images[i], axis=0)))].upper()}")
        plt.axis("off")
    break  # Afficher uniquement le premier batch
plt.show()
