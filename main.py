import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist
import numpy as np
import cv2

#Git

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_test)
print("")

model = keras.models.load_model("model/model.keras")

image = cv2.imread('data/test_img.png', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

resized_image = cv2.resize(image, (28, 28))

normalized_image = resized_image / 255.0

reshaped_image = normalized_image.reshape(1, 28, 28)

prediction = model.predict(reshaped_image)

# Get the predicted class (digit)
predicted_class = np.argmax(prediction)

print("Predicted class:", predicted_class)