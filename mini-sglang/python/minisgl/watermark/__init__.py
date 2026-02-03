"""Watermarking module for Mini-SGLang.

This module provides SynthID-Text watermarking capabilities for LLM text generation.
"""

from .logits_processor import WatermarkLogitsProcessor

__all__ = ["WatermarkLogitsProcessor"]
