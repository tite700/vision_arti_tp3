import numpy as np
import tensorflow as tf
import keras

layer = keras.layers.Dense(3)

layer.build((None, 4)) # Create the weight
layer.trainable = False

print("weights", len(layer.weights))
print("trainable_weights", len(layer.trainable_weights))
print("non_trainable_weights", len(layer.non_trainable_weights))

layer1 = keras.layers.Dense(3, activation="relu")
layer2 = keras.layers.Dense(3, activation="sigmoid")
model = keras.Sequential([
    keras.Input(3,),
    layer1,
    layer2
])

layer1.trainable = False

initial_layer1_weights_values = layer1.get_weights()

model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

final_layer1_weights_values = layer1.get_weights()

np.testing.assert_allclose(
    initial_layer1_weights_values[0], final_layer1_weights_values[0]
)
np.testing.assert_allclose(
    initial_layer1_weights_values[1], final_layer1_weights_values[1]
)