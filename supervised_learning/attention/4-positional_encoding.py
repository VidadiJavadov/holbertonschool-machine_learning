#!/usr/bin/env python3
"""
Positional Encoding for a Transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: An integer representing the maximum sequence length.
        dm: The model depth (dimensionality of the embeddings).

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    # Initialize the matrix
    PE = np.zeros((max_seq_len, dm))

    # Generate the positions (0 to max_seq_len - 1) -> shape (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Generate the division term for the formulas
    # 10000^(2i / dm) where i is the index of the dimension
    # We compute this for the even indices (2i)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply sine to even indices: sin(pos / 10000^(2i/dm))
    PE[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices: cos(pos / 10000^(2i/dm))
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
