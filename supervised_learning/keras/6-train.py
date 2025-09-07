#!/usr/bin/env python3
"""train model"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent with optional validation and early stopping."""
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               restore_best_weights=True)
        callbacks.append(early_stop)

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
