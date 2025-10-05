#!/usr/bin/env python3
"""lenet5"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.
    """
    he_normal = K.initializers.HeNormal(seed=0)

    # 1️⃣ Convolutional layer 1
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=he_normal)(X)

    # 2️⃣ Max pooling layer 1
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # 3️⃣ Convolutional layer 2
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=he_normal)(pool1)

    # 4️⃣ Max pooling layer 2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # 5️⃣ Flatten before dense layers
    flat = K.layers.Flatten()(pool2)

    # 6️⃣ Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=he_normal)(flat)

    # 7️⃣ Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=he_normal)(fc1)

    # 8️⃣ Output softmax layer with 10 nodes
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=he_normal)(fc2)

    # Build and compile the model
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
