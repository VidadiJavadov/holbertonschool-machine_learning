#!/usr/bin/env python3
"""optimize"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Configures Adam optimizer with categorical crossentropy and accuracy."""
    op = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=op,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
