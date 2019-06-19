from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image

import numpy as np


model = load_model('NeuralNetworkData.h5')
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_dir = r'C:\Users\Krystian Wojakiewicz\Desktop\Sieci\test'
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator)
print('dokladnosc podczas testowania:', test_acc)


test_image = image.load_img('sick5.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_acc = model.predict_classes(test_image)
print(test_acc)
if test_acc[0][0] == 1:
    prediction = 'sick'
else:
    prediction = 'healthy'

test_image = image.load_img('hel2.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_acc = model.predict_classes(test_image)
print(test_acc)
if test_acc[0][0] == 1:
    prediction = 'sick'
else:
    prediction = 'healthy'

