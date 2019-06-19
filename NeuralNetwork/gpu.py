from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
import matplotlib.pyplot as plt
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()


# optymalizator
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=-.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\REPO\ProjektZespolowyKWKWKS\Preprocessed'
train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(50, 50),
            batch_size=32,
            class_mode='binary')

validation_dir = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\REPO\ProjektZespolowyKWKWKS\val'
validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(50, 50),
            batch_size=32,
            class_mode='binary')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=50)

train_labels = ['healthy', 'sick']
#history = model.fit_generator(train_generator, train_labels, epochs=5)
# 40 epochs wystarczy"


model.save('a.h5')


# generowanie wykresow
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.legend()

plt.figure()

plt.plot(epochs, loss, ' bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()

plt.show()
