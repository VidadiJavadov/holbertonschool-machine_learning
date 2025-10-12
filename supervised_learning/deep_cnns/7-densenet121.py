#!/usr/bin/env python3
"""DenseNet-121 model"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, AveragePooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

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
    he_init = tf.keras.initializers.HeNormal(seed=0)
    
    # Input layer
    inputs = Input(shape=(224, 224, 3))
    
    # Initial convolution and pooling
    x = Conv2D(filters=64,
               kernel_size=7,
               strides=2,
               padding='same',
               kernel_initializer=he_init)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Dense Block 1
    x, channels = dense_block(x, 6, growth_rate)  # 6 layers in first block
    
    # Transition Layer 1
    x, channels = transition_layer(x, channels, compression)
    
    # Dense Block 2
    x, channels = dense_block(x, 12, growth_rate)  # 12 layers in second block
    
    # Transition Layer 2
    x, channels = transition_layer(x, channels, compression)
    
    # Dense Block 3
    x, channels = dense_block(x, 24, growth_rate)  # 24 layers in third block
    
    # Transition Layer 3
    x, channels = transition_layer(x, channels, compression)
    
    # Dense Block 4
    x, channels = dense_block(x, 16, growth_rate)  # 16 layers in fourth block
    
    # Global average pooling
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layer
    outputs = Dense(1000, activation='softmax', kernel_initializer=he_init)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
