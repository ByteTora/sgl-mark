"""Watermarking module for Mini-SGLang.

This module provides SynthID-Text watermarking capabilities for LLM text generation.
"""

from .scoring import mean_score, weighted_mean_score

__all__ = ["mean_score", "weighted_mean_score", "WatermarkLogitsProcessor"]

def __getattr__(name):
    if name == "WatermarkLogitsProcessor":
        from .logits_processor import WatermarkLogitsProcessor
        return WatermarkLogitsProcessor
    raise AttributeError(f"module {__name__} has no attribute {name}")
