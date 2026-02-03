"""Watermark logits processor for Mini-SGLang.

This module adapts SynthID-Text's watermarking algorithm to work with
Mini-SGLang's batched inference architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.utils import init_logger

# Import vendorized SynthID-Text components
try:
    from .vendor.logits_processing import (
        SynthIDLogitsProcessor,
    )
    SYNTHID_AVAILABLE = True
except ImportError:
    SYNTHID_AVAILABLE = False
    SynthIDLogitsProcessor = None

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.engine.sample import BatchSamplingArgs

logger = init_logger(__name__)


class WatermarkLogitsProcessor:
    """Watermarking logits processor adapter for Mini-SGLang.
    
    This class manages SynthID-Text watermarking for batched inference,
    maintaining separate watermark states for each request that has
    watermarking enabled.
    
    Attributes:
        device: PyTorch device for computations
        vocab_size: Vocabulary size of the model
        processors: Map from request UID to SynthIDLogitsProcessor instance
    """

    def __init__(
        self,
        device: torch.device,
        vocab_size: int,
    ):
        """Initialize the watermark logits processor.
        
        Args:
            device: PyTorch device to use for watermark computations
            vocab_size: Size of the model's vocabulary
        """
        if not SYNTHID_AVAILABLE:
            logger.warning(
                "synthid_text library not available. "
                "Watermarking will be disabled. "
                "Install with: pip install synthid-text"
            )
        
        self.device = device
        self.vocab_size = vocab_size
        # Map from request UID to SynthIDLogitsProcessor instance
        self.processors: Dict[int, SynthIDLogitsProcessor] = {}
        
    def _create_processor(
        self,
        keys: List[int],
        ngram_len: int,
        context_history_size: int,
        temperature: float,
        top_k: int,
    ) -> SynthIDLogitsProcessor:
        """Create a new SynthIDLogitsProcessor instance.
        
        Args:
            keys: Watermarking keys (one per depth layer)
            ngram_len: N-gram context length
            context_history_size: Size of context history buffer
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Configured SynthIDLogitsProcessor instance
        """
        if not SYNTHID_AVAILABLE:
            raise RuntimeError(
                "Cannot create watermark processor: synthid_text not installed"
            )
            
        return SynthIDLogitsProcessor(
            ngram_len=ngram_len,
            keys=keys,
            context_history_size=context_history_size,
            temperature=temperature,
            top_k=top_k,
            device=self.device,
            skip_first_ngram_calls=False,
            apply_top_k=True,
            num_leaves=30,  # Increased for much stronger injection signal
        )
    
    def process_batch(
        self,
        batch: Batch,
        logits: torch.Tensor,
        sample_args: BatchSamplingArgs,
    ) -> torch.Tensor:
        """Process logits for a batch, applying watermarking where enabled.
        
        This method handles mixed batches where some requests have watermarking
        enabled and others don't. It applies watermarking only to requests that
        have watermark_enabled=True in their sampling parameters.
        
        Args:
            batch: Current batch of requests
            logits: Model output logits [batch_size, vocab_size]
            sample_args: Batch sampling arguments with temperature/top-k
            
        Returns:
            Processed logits tensor [batch_size, vocab_size]
        """
        if not SYNTHID_AVAILABLE:
            # If SynthID-Text is not available, return logits unchanged
            return logits
        
        # Check if any request in the batch has watermarking enabled
        watermark_enabled_mask = [
            req.sampling_params.watermark_enabled
            for req in batch.reqs
        ]
        
        if not any(watermark_enabled_mask):
            # No watermarking needed for this batch
            return logits
        
        # logger.info(f"[WATERMARK] Processing batch with {sum(watermark_enabled_mask)}/{len(watermark_enabled_mask)} watermarked requests")
        
        # Process each request individually
        processed_logits = logits.clone()
        
        for i, req in enumerate(batch.reqs):
            if not req.sampling_params.watermark_enabled:
                continue
                
            # Get or create processor for this request
            if req.uid not in self.processors:
                # Validate watermark configuration
                if req.sampling_params.watermark_keys is None:
                    logger.warning(
                        f"Request {req.uid} has watermark_enabled=True but "
                        "no watermark_keys provided. Skipping watermarking."
                    )
                    continue
                
                # Create new processor
                try:
                    self.processors[req.uid] = self._create_processor(
                        keys=req.sampling_params.watermark_keys,
                        ngram_len=req.sampling_params.watermark_ngram_len,
                        context_history_size=req.sampling_params.watermark_context_history_size,
                        temperature=max(req.sampling_params.temperature, 1e-6),
                        top_k=max(req.sampling_params.top_k, 1) if req.sampling_params.top_k > 0 else self.vocab_size,
                    )
                    logger.debug(f"[WATERMARK] Created processor for request {req.uid} with keys={req.sampling_params.watermark_keys[:3]}...")
                    logger.debug(f"[WATERMARK] Request {req.uid} parameters: ngram_len={req.sampling_params.watermark_ngram_len}, num_leaves=4")
                except Exception as e:
                    logger.error(
                        f"Failed to create watermark processor for request {req.uid}: {e}"
                    )
                    continue
            
            processor = self.processors[req.uid]
            
            # CRITICAL FIX: In Mini-SGLang, the input_ids in the request object 
            # are NOT updated on the engine side during decoding. 
            # We must pass only the NEWLY sampled token to the state 
            # or rely on the processor's internal state tracking.
            
            # If this is the first call (prefill), input_ids are the prompt.
            # If it's a decode call, we only care about the last token that was just added.
            
            request_logits = logits[i:i+1, :]
            
            try:
                # We need input_ids to initialize the very first context (prompt) 
                # but during decoding, the vendor code expects just the latest token.
                # In our case, we'll use a trick: 
                # 1. On first call: Pass prompt
                # 2. On subsequent calls: Pass only the last generated token
                
                if processor.state is None:
                    # Initial call: use prefix
                    current_input = req.input_ids[:req.device_len].unsqueeze(0).to(self.device)
                else:
                    # Subsequent calls: We need to know what we sampled last time.
                    # This is tricky because process_batch is called BEFORE the next sample.
                    # We'll store the results of sampling in our processor.
                    last_token = getattr(processor, '_last_sampled_token', None)
                    if last_token is not None:
                        current_input = torch.tensor([[last_token]], device=self.device)
                    else:
                        # Fallback (shouldn't happen often)
                        current_input = req.input_ids[:req.device_len].unsqueeze(0).to(self.device)

                # Apply watermarking
                watermarked_scores, top_k_indices, _ = processor.watermarked_call(
                    input_ids=current_input,
                    scores=request_logits,
                )
                
                # Reconstruct full logits
                full_watermarked_logits = torch.full_like(request_logits, -1e12, device=self.device)
                full_watermarked_logits.scatter_(dim=1, index=top_k_indices, src=watermarked_scores)
                processed_logits[i:i+1, :] = full_watermarked_logits
                
            except Exception as e:
                logger.error(f"Failed to apply watermarking for request {req.uid}: {e}")
                continue
        
        return processed_logits

    def update_last_token(self, batch: Batch, next_tokens: torch.Tensor):
        """Update the internal history with the newly sampled tokens."""
        for i, req in enumerate(batch.reqs):
            if req.uid in self.processors:
                # Store it so the NEXT step's process_batch knows what we picked
                self.processors[req.uid]._last_sampled_token = next_tokens[i].item()
    
    def cleanup_request(self, uid: int):
        """Clean up watermark processor for a finished request.
        
        This should be called when a request is completed to free memory.
        
        Args:
            uid: Request UID to clean up
        """
        if uid in self.processors:
            del self.processors[uid]
            logger.debug(f"Cleaned up watermark processor for request {uid}")
    
    def get_active_watermarks(self) -> int:
        """Get the number of active watermark processors.
        
        Returns:
            Number of requests currently being watermarked
        """
        return len(self.processors)
