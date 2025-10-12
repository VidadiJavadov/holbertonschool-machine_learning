#!/usr/bin/env python3
"""Builds the DenseNet-121 architecture as described in DenseNet."""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture."""
    he_normal = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate  # Initial number of filters

    # Initial convolution + max pooling
    X = K.layers.Conv2D(
        nb_filters, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=he_normal
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.GlobalAveragePooling2D()(X)

    # Output layer (ImageNet 1000 classes)
    X = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=he_normal
    )(X)

    model = K.Model(inputs=X_input, outputs=X)
    return model
