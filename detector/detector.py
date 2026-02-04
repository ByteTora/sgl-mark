"""Offline Watermark Detector for Mini-SGLang + SynthID-Text"""
import sys
import os
import argparse
import transformers
import torch
import numpy as np

# Setup paths to find dependencies
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # mini-sglang-final

# Add mini-sglang/python to sys.path to allow importing minisgl
minisgl_python_path = os.path.join(project_root, 'mini-sglang', 'python')
if os.path.exists(minisgl_python_path):
    sys.path.insert(0, minisgl_python_path)
else:
    print(f"Warning: minisgl python path not found at {minisgl_python_path}")

try:
    from minisgl.watermark.vendor import hashing_function, logits_processing
    from minisgl.watermark import mean_score
except ImportError as e:
    print(f"Error: Dependencies missing. {e}")
    sys.exit(1)

# Default configuration (matching typical server settings)
DEFAULT_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29]
NGRAM_LEN = 5
TOP_K = 200
TEMPERATURE = 1.0 # Neutral temperature for scoring

def analyze_watermark(text, keys, model_name="Qwen/Qwen3-0.6B", prompt_text=""):
    """Core detection logic matching the stateful engine behavior."""
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Pre-process text: we only detect watermark in the generated response part
    # If full_text starts with prompt_text, we isolate the response
    if prompt_text and text.startswith(prompt_text):
        response_text = text[len(prompt_text):]
    else:
        response_text = text

    # Tokenize full text
    full_text = prompt_text + response_text
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    # Calculate response token IDs
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if prompt_text else []
    prompt_len = len(prompt_ids)
    response_tokens = tokens[prompt_len:]
    
    if not response_tokens:
        return None

    # Init processor to get access to G-value calculation
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        ngram_len=NGRAM_LEN,
        keys=keys,
        context_history_size=1024,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        device=torch.device('cpu'),
    )

    # REPLICATE ENGINE BEHAVIOR:
    # The engine starts with context [0, 0, 0, 0] for the first token
    server_ngrams = []
    current_context = [0] * (NGRAM_LEN - 1)
    for t in response_tokens:
        server_ngrams.append(current_context + [t])
        current_context = (current_context + [t])[1:]
    
    # Compute G-values for all tokens in response
    ngrams_tensor = torch.tensor([server_ngrams], dtype=torch.int64)
    ngram_keys = logits_processor.compute_ngram_keys(ngrams_tensor)
    g_values = logits_processor.get_gvals(ngram_keys) # [1, L_resp, depth]
    g_values_np = g_values.cpu().numpy()
    
    # Create Mask
    L_resp = len(response_tokens)
    final_mask = np.ones((1, L_resp), dtype=bool)
    
    # Mask out first 20 tokens (warm-up/high entropy phase)
    SAFE_BUFFER = 20
    if L_resp > SAFE_BUFFER:
        final_mask[:, :SAFE_BUFFER] = False
    
    # Repetition Mask: exclude repeated contexts
    # We prefix with the same [0]*4 to match stateful hashing
    input_for_repo = torch.tensor([([0]*(NGRAM_LEN-1)) + response_tokens], dtype=torch.int64)
    repetition_mask = logits_processor.compute_context_repetition_mask(input_for_repo)
    combined_mask = final_mask & repetition_mask.cpu().numpy()
    
    # Handle EOS
    if tokenizer.eos_token_id is not None:
        for i, tid in enumerate(response_tokens):
            if tid == tokenizer.eos_token_id:
                combined_mask[:, i:] = False
                break

    valid_count = int(np.sum(combined_mask))
    if valid_count == 0:
        return {"mean_score": 0.0, "z_score": 0.0, "valid_tokens": 0}

    # Calculate Mean Score using local implementation
    score = mean_score(g_values_np, combined_mask)
    mean_score_val = float(score[0])
    
    # Standard Z-score for frequentist reporting
    # Formula: (mean - 0.5) / (std / sqrt(N * depth)) where std=0.5
    n_total = valid_count * len(keys)
    z_score = (mean_score_val - 0.5) * np.sqrt(12 * n_total)
    
    return {
        "mean_score": mean_score_val,
        "z_score": z_score,
        "valid_tokens": valid_count
    }

def main():
    parser = argparse.ArgumentParser(description="SynthID-Text Watermark Detector")
    parser.add_argument("--input", required=True, help="Path to the text file to detect")
    parser.add_argument("--keys", required=True, help="Comma-separated integers for watermark keys")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name used for tokenization")
    parser.add_argument("--prompt", help="Original prompt text (optional, helps precision)")
    
    args = parser.parse_args()
    
    try:
        keys = [int(k.strip()) for k in args.keys.split(",")]
    except ValueError:
        print("Error: Keys must be a list of comma-separated integers.")
        sys.exit(1)
        
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        sys.exit(1)
        
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
        
    prompt = ""
    if args.prompt and os.path.exists(args.prompt):
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt

    print("=" * 70)
    print("SynthID-Text Watermark Detection")
    print("=" * 70)
    print(f"Analyzing: {os.path.basename(args.input)}")
    
    result = analyze_watermark(text, keys, args.model, prompt)
    
    if result:
        print("\nResults:")
        print(f"  Mean Score:   {result['mean_score']:.6f}")
        print(f"  Valid Tokens: {result['valid_tokens']}")
        print(f"  Z-Score:      {result['z_score']:.4f}")
        print("\nVerdict:")
        if result['z_score'] > 4.0:
            print("  ✅ WATERMARK DETECTED (High Confidence)")
        elif result['z_score'] > 2.0:
            print("  ⚠️ WEAK SIGNAL DETECTED")
        else:
            print("  ❌ NO WATERMARK DETECTED")
    else:
        print("\nError: Could not analyze text.")
    print("=" * 70)

if __name__ == "__main__":
    main()
