import keras
import numpy as np
from keras.layers import Rescaling
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

'''
Simple transfert learning model
'''

# Définir le chemin vers votre dossier d'images
data_dir = "./data/natural_images"

# Définir la taille des images en entrée de VGG16 (224x224 pixels)
img_size = (224, 224)

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


def show_dataset(ds):
    # Visualize dataset
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            index = np.argmax(labels[i])
            plt.title(class_names[index])
            plt.axis('off')

    plt.show()


# Visualize dataset
show_dataset(train_ds)

# Base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

transfert_learning_model = Sequential([
    Rescaling(scale=1.0/127.5, offset=-1),
    base_model,
    Flatten(),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(num_class, activation='softmax')
])

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    custom_cnn_model = Sequential([
        Conv2D(6, kernel_size=5, activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=5, activation='relu'),
        Conv2D(16, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(units=120, activation='relu'),
        Dense(units=84, activation='relu'),
        Dense(num_class, activation='softmax')
    ])
    transfert_learning_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    custom_cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the transfert learning model
    transfert_learning_model_history = transfert_learning_model.fit(train_ds, epochs=5, validation_data=val_ds)
    # Train the custom cnn model
    custom_cnn_model_history = custom_cnn_model.fit(train_ds, epochs=5, validation_data=val_ds)

# Transfert learning
transfert_learning_predictions = transfert_learning_model.predict(test_ds)
transfert_learning_pred_classes = np.argmax(transfert_learning_predictions, axis=1)

# Custom cnn
custom_cnn_predictions = custom_cnn_model.predict(test_ds)
custom_cnn_pred_classes = np.argmax(custom_cnn_predictions, axis=1)

# Function to visualize the prediction of the model
def visualize_prediction(prediction, fig_name):
    plt.figure(figsize=(10, 10))
    for images, labels in test_ds:
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            index_class = np.argmax(labels[i])
            index_prediction = np.argmax(prediction[i])
            plt.title(f"class={class_names[index_class]}, prediction={class_names[index_prediction]}")
            plt.axis('off')

    plt.savefig(fig_name + '.jpg')
    plt.show()


# Visualize the transfert learning model prediction
visualize_prediction(transfert_learning_predictions, 'transfert-learning')

# Visualize the custom cnn model prediction
visualize_prediction(custom_cnn_predictions, 'custom-cnn')

# Calculate the accuracy on the test dataset
true_classes = []
for images, labels in test_ds:
    for i in range(len(images)):
        true_classes.append(np.argmax(labels[i]))

# Transfert learning model statistics
transfert_learning_accuracy = accuracy_score(true_classes, transfert_learning_pred_classes)
transfert_learning_kappa_score = cohen_kappa_score(true_classes, transfert_learning_pred_classes)
transfert_learning_report = classification_report(true_classes, transfert_learning_pred_classes)

custom_cnn_accuracy = accuracy_score(true_classes, custom_cnn_pred_classes)
custom_cnn_kappa_score = cohen_kappa_score(true_classes, custom_cnn_pred_classes)
custom_cnn_report = classification_report(true_classes, custom_cnn_pred_classes)

print("Transfer Learning (VGG16) - Accuracy:", transfert_learning_accuracy, "Kappa:", transfert_learning_kappa_score)
print("Transfer Learning (VGG16) Report:\n", transfert_learning_report)

print("Custom CNN - Accuracy:", custom_cnn_accuracy, "Kappa:", custom_cnn_kappa_score)
print("Custom CNN Report:\n", custom_cnn_report)

# Courbe d'apprentissage du modèle de transfer learning avec VGG16
plt.subplot(1, 1, 1)
plt.plot(transfert_learning_model_history.history['accuracy'], label='Training Accuracy')
plt.plot(transfert_learning_model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning (VGG16) - Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('TransfertLearning.jpg')
plt.show()

# Courbe d'apprentissage du modèle de transfer learning avec VGG16
plt.subplot(1, 1, 1)
plt.plot(custom_cnn_model_history.history['accuracy'], label='Training Accuracy')
plt.plot(custom_cnn_model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Custom CNN - Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('CustomCNN.jpg')
plt.show()