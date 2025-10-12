#!/usr/bin/env python3
"""DenseNet-121 model"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer

def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture
    
    Args:
        growth_rate (int): growth rate for the dense blocks
        compression (float): compression factor for transition layers
    Returns:
        keras.Model: DenseNet-121 model
    """
    he_init = K.initializers.HeNormal(seed=0)
    
    # Input layer
    inputs = K.Input(shape=(224, 224, 3))
    
    # Initial BatchNorm + ReLU + Conv + MaxPool
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.ReLU()(x)
    x = K.layers.Conv2D(64, kernel_size=7, strides=2, padding='same',
                        kernel_initializer=he_init)(x)
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Dense Block 1
    x, channels = dense_block(x, 6, growth_rate)
    
    # Transition Layer 1
    x, channels = transition_layer(x, channels, compression)
    
    # Dense Block 2
    x, channels = dense_block(x, 12, growth_rate)
    
    # Transition Layer 2
    x, channels = transition_layer(x, channels, compression)
    
    # Dense Block 3
    x, channels = dense_block(x, 24, growth_rate)
    
    # Transition Layer 3
    x, channels = transition_layer(x, channels, compression)
    
    # Dense Block 4
    x, channels = dense_block(x, 16, growth_rate)
    
    # Final BatchNorm + ReLU + GlobalAvgPool
    x = K.layers.BatchNormalization()(x)
    x = K.layers.ReLU()(x)
    x = K.layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=he_init)(x)
    
    # Create model
    model = K.Model(inputs=inputs, outputs=outputs)
    
    return model
