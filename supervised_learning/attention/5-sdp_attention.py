#!/usr/bin/env python3
"""
Scaled Dot Product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q: Tensor (..., seq_len_q, dk) containing the query matrix.
        K: Tensor (..., seq_len_v, dk) containing the key matrix.
        V: Tensor (..., seq_len_v, dv) containing the value matrix.
        mask: Tensor that can be broadcast into (..., seq_len_q, seq_len_v)
              containing the optional mask, or defaulted to None.

    Returns:
        output, weights
        output: Tensor (..., seq_len_q, dv) containing the scaled dot product
                attention.
        weights: Tensor (..., seq_len_q, seq_len_v) containing the attention
                 weights.
    """
    # 1. Matmul Q and K (transpose K to match dimensions)
    # Shape: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # 2. Scale by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 3. Apply mask if it exists
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 4. Softmax to get weights
    # Shape: (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 5. Weighted sum of V
    # Shape: (..., seq_len_q, dv)
    output = tf.matmul(weights, V)

    return output, weights
