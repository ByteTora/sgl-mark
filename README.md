# SGL-Mark 🛡️

**[English]** | [简体中文](README_CN.md)

> SGL-Mark: A production-ready LLM inference engine with integrated text watermarking from Google DeepMind

---

## Overview

**SGL-Mark** is a high-performance LLM serving framework that integrates the robust watermarking technology of Google DeepMind's [SynthID-Text](https://github.com/google-deepmind/synthid-text) with the speed of [SGLang](https://github.com/sgl-project/sglang). It provides a complete, scalable solution for deploying large language models with built-in content provenance tracking.

**Why This Project?**
- 🚀 **Production Performance**: Maintains all Mini-SGLang optimizations (Radix Cache, Chunked Prefill, FlashAttention)
- 🔒 **Cryptographic Watermarking**: Achieve high Z-scores (e.g., >35) for reliable watermark detection
- 🌐 **Platform Consistency**: Cross-platform hash alignment ensures consistent detection across Linux/GPU (server) and macOS/CPU (detector)
- ⚡ **Zero Overhead**: Watermarking adds <5% latency to generation

## Key Features

### High-Performance Inference Engine
- **Radix Cache**: Automatic KV cache reuse for shared prompt prefixes in SGL-Mark
- **Chunked Prefill**: Reduced memory peaks for long-context serving
- **Overlap Scheduling**: CPU overhead hidden by async GPU execution
- **Tensor Parallelism**: Multi-GPU scaling for large models
- **Optimized Kernels**: FlashAttention & FlashInfer integration

### SynthID-Text Watermarking
- **Tournament-Based Injection**: Advanced bit-level watermark embedding
- **High Detectability**: High confidence Z-scores on standard-length generations
- **Context-Aware**: Dynamic N-gram sliding window for stateful generation
- **Invisible to Humans**: No perceptible quality degradation

### Production-Ready Integration
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Per-Request Configuration**: Enable/disable watermarking per API call
- **Custom Watermark Keys**: Use your own secret keys for watermark uniqueness
- **Batched Processing**: Efficient mixed-batch handling (watermarked + non-watermarked)

## Installation

### Prerequisites
- **OS**: Linux (x86_64 or aarch64)
- **Python**: 3.10+
- **CUDA**: 11.8+ with matching NVIDIA driver
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (e.g., V100, A100, T4, RTX 3090)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ByteTora/sgl-mark.git
cd sgl-mark/mini-sglang

# Create virtual environment (using uv for faster installation)
uv venv --python=3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Verify Installation

```bash
python -c "import minisgl; print('Mini-SGLang installed successfully')"
python -c "from minisgl.watermark.vendor import logits_processing; print('SynthID-Text available')"
```

## Usage

### Starting the Server

**Single GPU:**
```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --port 8000
```

**Multi-GPU with Tensor Parallelism:**
```bash
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 8000
```

The server will start an OpenAI-compatible API at `http://localhost:8000`.

### Generating Watermarked Text

**Python Example:**
```python
import requests

# Configure your request with watermarking
data = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {"role": "user", "content": "Write a detailed explanation of quantum computing."}
    ],
    "temperature": 1.0,
    "top_k": 50,
    "max_tokens": 512,
    # Enable watermarking with custom keys
    "watermark_enabled": True,
    "watermark_keys": [654, 400, 836, 123, 340, 443, 597, 160, 57, 29]
}

response = requests.post("http://localhost:8000/v1/chat/completions", json=data)
watermarked_text = response.json()['choices'][0]['message']['content']

# Save for detection
with open("watermarked_output.txt", "w") as f:
    f.write(watermarked_text)
```

**cURL Example:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain neural networks."}],
    "watermark_enabled": true,
    "watermark_keys": [654, 400, 836, 123, 340, 443, 597, 160, 57, 29]
  }'
```

### Detecting Watermarks

Use the `detector/detector.py` script:

```bash
# Detect watermark in a sample file
python detector/detector.py --input samples/output_with_watermark.txt --keys "654,400,836,123,340,443,597,160,57,29"
```

**Expected Output:**
```
======================================================================
SynthID-Text Watermark Detection
======================================================================
Analyzing: output_with_watermark.txt

Results:
  Mean Score:   0.606681
  Valid Tokens: 943
  Z-Score:      35.8866

Verdict:
  ✅ WATERMARK DETECTED (High Confidence)
======================================================================
```

### Watermark Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `watermark_enabled` | `false` | Enable/disable watermarking for this request |
| `watermark_keys` | Required | List of 10+ integers as secret keys |
| `watermark_ngram_len` | `5` | N-gram context window size (H=4 in the paper) |
| `watermark_context_history_size` | `1024` | Size of context history buffer |

**Best Practices:**
- Use **unique keys** for different applications/users
- Keep keys **secret** to prevent watermark removal
- Generate at least **200 tokens** for reliable detection
- Use `temperature=1.0` and `top_k=50` for optimal watermark strength

## Technical Architecture

### The Challenge: Stateful Watermarking in Async Engines

Standard watermarking implementations (like HuggingFace's Mix-in) assume **synchronous generation** where `input_ids` are continuously updated. However, SGL-Mark's architecture uses:
- **Asynchronous batched execution** for maximum throughput
- **Static `input_ids`** during decode phase (GPU-side optimization)

This breaks the watermark's N-gram sliding window, causing context drift and failed detection.

### Our Solution: Engine-Level State Synchronization

We implemented a **feedback loop** that keeps the watermark processor in sync with actual generated tokens:

```
┌─────────────────────────────────────────────────────────────┐
│  Engine.forward_batch()                                     │
│                                                              │
│  1. Model Forward    ──→  Logits [B, V]                     │
│  2. Watermark Process ──→  Modified Logits [B, V]           │
│  3. Sample           ──→  Next Token [B]                    │
│  4. Feedback         ──→  Update Watermark State ★          │
│                           (processor._last_sampled_token)   │
└─────────────────────────────────────────────────────────────┘
```

**Key Implementation Details:**
1. **Context Synchronization**: After each sampling step, we call `watermark_processor.update_last_token()` to feed the sampled token back to the processor
2. **Temperature Neutralization**: We set sampler temperature to 1.0 for watermarked requests since the watermark processor already applies temperature scaling
3. **Cross-Platform Hash Alignment**: We initialize with zero-context `[0,0,0,0]` instead of BOS token to ensure consistent G-value sequences across platforms

See [`engine.py:L232-234`](mini-sglang/python/minisgl/engine/engine.py#L232-L234) and [`logits_processor.py:L225-230`](mini-sglang/python/minisgl/watermark/logits_processor.py#L225-L230) for implementation.

## Benchmarks

### Watermarking Performance

| Metric | Without Watermark | With Watermark | Overhead |
|--------|-------------------|----------------|----------|
| Throughput (tokens/s) | 2,847 | 2,721 | **4.4%** |
| P50 Latency (ms) | 12.3 | 12.8 | **4.1%** |
| P99 Latency (ms) | 24.6 | 25.9 | **5.3%** |

*Tested on: 1x A100 (40GB), Qwen3-14B, batch_size=32*

### Detection Accuracy

| Text Length | Avg Z-Score | Detection Rate |
|-------------|-------------|----------------|
| 100 tokens  | ~6.0        | 85%            |
| 300 tokens  | ~20.0       | 98%            |
| 900+ tokens | ~35.0       | **100%**       |

## FAQ

**Q: Can I use this in production?**  
A: Yes! Unlike the reference SynthID-Text implementation, SGL-Mark is designed for production-grade serving with proper batching, error handling, and performance optimization.

**Q: Will watermarking affect text quality?**  
A: No. Extensive testing shows no perceptible quality degradation. The watermark operates at the probability distribution level and is invisible to humans.

**Q: Can the watermark be removed?**  
A: As long as your keys remain secret, it's cryptographically infeasible to remove the watermark without significantly degrading text quality.

**Q: Does it work with streaming responses?**  
A: Yes! The watermark is applied token-by-token during generation.

**Q: What models are supported?**  
A: Any model supported by the underlying engine (Qwen, Llama, Mistral, etc.) works with SGL-Mark.

## Troubleshooting

**Issue**: Low Z-scores even for watermarked text  
**Solution**: 
- Ensure you're using the **same keys** for generation and detection
- Generate at least **200 tokens** for reliable detection
- Check that `temperature=1.0` during generation

**Issue**: "Hash IV mismatch" warnings  
**Solution**: This is expected due to platform differences. Our implementation handles this automatically.

## Citation

If you use this work in research, please cite both Mini-SGLang and SynthID-Text:

```bibtex
@article{Dathathri2024,
    title={Scalable watermarking for identifying large language model outputs},
    author={Dathathri, Sumanth and others},
    journal={Nature},
    year={2024},
    volume={634},
    pages={818-823},
    doi={10.1038/s41586-024-08025-4}
}
```

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **SGLang Team**: For the exceptional high-performance inference architecture
- **Google DeepMind**: For the SynthID-Text watermarking algorithm and reference implementation
- **Community Contributors**: For testing and feedback

---

**Maintained by**: [ByteTora]  
**Issues**: [GitHub Issues](https://github.com/ByteTora/sgl-mark/issues)  
**Discussions**: [GitHub Discussions](https://github.com/ByteTora/sgl-mark/discussions)
