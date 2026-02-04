# Watermark Detector 🕵️‍♂️

This module provides the offline detection capabilities for the **SGL-Mark** project. It allows you to verify if a text was generated with a specific set of watermark keys.

## Features

- **Stateful Alignment**: Perfectly replicates the SGLang engine's generation behavior (starting context, G-value sequences).
- **Z-Score Reporting**: Provides a standard statistical confidence score.
- **Masking Aware**: Corrects for context repetitions and high-entropy warm-up tokens.
- **Cross-Platform**: Designed to run on CPU-only environments (e.g., macOS/Linux).

## Installation

Ensure you have the main project dependencies installed:

```bash
# From the project root
pip install -e ./mini-sglang
```

The detector uses **NumPy** for scoring, making it lightweight and compatible with standard CPU environments.

## Usage

Run the detection script by providing the text file and the keys used during generation.

Run the detection script by providing the text file and the keys used during generation.

```bash
# Example using the provided sample
python detector.py --input ../samples/output_with_watermark.txt --keys "654,400,836,123,340,443,597,160,57,29"
```

### Options

| Flag | Description |
|------|-------------|
| `--input` | Path to the text to be analyzed (Required) |
| `--keys` | Comma-separated list of 10+ integers (Secret Keys) |
| `--model` | The tokenizer name (default: `Qwen/Qwen3-0.6B`) |
| `--prompt` | Path to a file containing the original prompt (Optional, improves precision) |

## How Detection Works

The detector performs the following steps:
1. **Tokenization**: Converts text into IDs using the same tokenizer as the server.
2. **Context Synchronization**: Simulates the exact stateful generation path used by the SGLang engine.
3. **G-Value Reconstruction**: Calculates the binary watermark signals (G-values) for every token position.
4. **Statistical Scoring**: Applies a weighted mean score and calculates the Z-Score based on the deviation from random noise (0.5 benchmark).

### Interpreting Z-Scores

- **Z-Score > 4.0**: High confidence watermark detected.
- **Z-Score > 10.0**: Extremely high confidence (almost zero false positives).
- **Z-Score < 1.0**: Random noise or unwatermarked text.

## Technical Notes

Our detector is specifically optimized to match the **SGL-Mark Engine**. Common issues with generic SynthID detectors (like BOS vs. Zero-context mismatch) are resolved here by aligning the initial context window to `[0, 0, 0, 0]`.
