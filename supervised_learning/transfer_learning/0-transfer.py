import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau


from tensorflow.keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)

train_images = tf.image.resize(train_images, [96, 96])
test_images = tf.image.resize(test_images, [96, 96])

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(train_images)


base_model = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

inputs = Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(datagen.flow(train_images, train_labels, batch_size=32), 
          epochs=10, 
          validation_data=(test_images, test_labels))


base_model.trainable = True
for layer in base_model.layers[:100]:
  layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),   # 👈 still use generator
    epochs=20,
    validation_data=(test_images, test_labels),
    callbacks=[lr_scheduler] 
)
