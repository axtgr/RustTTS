#!/usr/bin/env python3
"""
Generate reference audio and tokens using Qwen3-TTS Python SDK.
This creates golden files for comparing with Rust implementation.
"""

import json
import sys
from pathlib import Path

import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
MODEL_PATH = "models/qwen3-tts-0.6b-customvoice"
TEXT = "Hello world"
SPEAKER = "ryan"
LANG = "english"
OUTPUT_DIR = Path("tests/golden")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )

    print(f"Model loaded: {type(model)}")
    print(f"Model config: {model.config}")

    # Check if model has generate_speech or similar method
    print(
        f"\nModel methods containing 'generate': {[m for m in dir(model) if 'generate' in m.lower()]}"
    )
    print(
        f"Model methods containing 'speech': {[m for m in dir(model) if 'speech' in m.lower()]}"
    )
    print(
        f"Model methods containing 'tts': {[m for m in dir(model) if 'tts' in m.lower()]}"
    )

    # Try to understand the model architecture
    print(f"\nModel children:")
    for name, child in model.named_children():
        print(f"  {name}: {type(child)}")

    # Get special tokens
    print(f"\nTokenizer special tokens:")
    print(f"  vocab_size: {tokenizer.vocab_size}")

    # Try to find TTS-specific tokens
    special_tokens = {}
    for name in [
        "tts_bos",
        "tts_eos",
        "tts_pad",
        "codec_bos",
        "codec_eos",
        "codec_pad",
    ]:
        token_id = getattr(tokenizer, f"{name}_token_id", None)
        if token_id is not None:
            special_tokens[name] = token_id
    print(f"  TTS tokens: {special_tokens}")

    # Tokenize input text
    print(f"\nTokenizing: '{TEXT}'")
    tokens = tokenizer.encode(TEXT, add_special_tokens=False)
    print(f"  Token IDs: {tokens}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in tokens]}")

    # Save tokenization info
    info = {
        "text": TEXT,
        "speaker": SPEAKER,
        "lang": LANG,
        "text_tokens": tokens,
        "special_tokens": special_tokens,
        "model_type": str(type(model)),
    }

    with open(OUTPUT_DIR / "reference_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved info to {OUTPUT_DIR / 'reference_info.json'}")

    # Try to generate if possible
    # This depends on the model having a specific generation method


if __name__ == "__main__":
    main()
