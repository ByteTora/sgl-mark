"""Scoring functions for SynthID-Text watermark detection."""

import numpy as np
from typing import Optional

def mean_score(
    g_values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Computes the Mean score using NumPy.

    Args:
        g_values: g-values of shape [batch_size, seq_len, watermarking_depth].
        mask: A binary array shape [batch_size, seq_len] indicating which g-values
            should be used. g-values with mask value 0 are discarded.

    Returns:
        Mean scores, of shape [batch_size]. This is the mean of the unmasked
            g-values.
    """
    watermarking_depth = g_values.shape[-1]
    num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
    
    # Avoid division by zero
    num_unmasked = np.maximum(num_unmasked, 1e-9)
    
    return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
        watermarking_depth * num_unmasked
    )

def weighted_mean_score(
    g_values: np.ndarray,
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes the Weighted Mean score using NumPy.

    Args:
        g_values: g-values of shape [batch_size, seq_len, watermarking_depth].
        mask: A binary array shape [batch_size, seq_len] indicating which g-values
            should be used. g-values with mask value 0 are discarded.
        weights: array of non-negative floats, shape [watermarking_depth]. The
            weights to be applied to the g-values. If not supplied, defaults to
            linearly decreasing weights from 10 to 1.

    Returns:
        Weighted Mean scores, of shape [batch_size]. This is the mean of the
            unmasked g-values, re-weighted using weights.
    """
    watermarking_depth = g_values.shape[-1]

    if weights is None:
        weights = np.linspace(start=10, stop=1, num=watermarking_depth)

    # Normalise weights so they sum to watermarking_depth.
    weights = weights * (watermarking_depth / np.sum(weights))

    # Apply weights to g-values.
    # weights is [depth], g_values is [batch, seq, depth]
    weighted_g_values = g_values * weights

    num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
    
    # Avoid division by zero
    num_unmasked = np.maximum(num_unmasked, 1e-9)
    
    return np.sum(weighted_g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
        watermarking_depth * num_unmasked
    )
