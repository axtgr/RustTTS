#!/usr/bin/env python3
"""
Generate reference audio codes using GREEDY decoding (no sampling).
This ensures deterministic output for comparison with Rust implementation.
"""

import json
import sys
from pathlib import Path

import torch

# Add Qwen3-TTS to path
sys.path.insert(0, "/tmp/Qwen3-TTS")

# Config
MODEL_PATH = "models/qwen3-tts-0.6b-customvoice"
TEXT = "Hello world"
SPEAKER = "ryan"
LANG = "english"
OUTPUT_DIR = Path("tests/golden")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")

    # Import Qwen3-TTS components
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models import Qwen3TTSProcessor

    # Load config
    config = Qwen3TTSConfig.from_pretrained(MODEL_PATH)
    print(f"Config loaded: {config.tts_model_type}")

    # Load model
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Load processor/tokenizer
    processor = Qwen3TTSProcessor.from_pretrained(MODEL_PATH)

    print(f"Model loaded successfully")

    # Build input text in CustomVoice format
    assistant_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\nInput text: {assistant_text!r}")

    # Tokenize
    inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")

    # Generate with GREEDY decoding (do_sample=False)
    print("\nGenerating with GREEDY decoding (deterministic)...")
    with torch.no_grad():
        talker_codes_list, hidden_states_list = model.generate(
            input_ids=[input_ids],
            instruct_ids=None,
            languages=[LANG],
            speakers=[SPEAKER],
            non_streaming_mode=True,
            max_new_tokens=512,
            do_sample=False,  # GREEDY - always pick argmax
        )

    print(f"Generated {len(talker_codes_list)} sequences")

    if talker_codes_list:
        codes = talker_codes_list[0]
        print(f"Codes shape: {codes.shape}")
        print(f"First 20 frames (zeroth codebook):")
        for i, frame in enumerate(codes[:20].tolist()):
            print(f"  Frame {i}: {frame[0]} (all: {frame})")
        print(f"\nTotal frames: {len(codes)}")

        # Extract zeroth codebook (first element of each frame)
        zeroth = [frame[0] for frame in codes.tolist()]
        print(f"\nZeroth codebook tokens: {zeroth}")

        # Save codes
        codes_path = OUTPUT_DIR / "reference_codes_greedy.json"
        with open(codes_path, "w") as f:
            json.dump(
                {
                    "text": TEXT,
                    "speaker": SPEAKER,
                    "language": LANG,
                    "input_ids": input_ids.tolist(),
                    "codes": codes.tolist(),
                    "codes_shape": list(codes.shape),
                    "zeroth_codebook": zeroth,
                    "decoding": "greedy",
                },
                f,
                indent=2,
            )
        print(f"\nSaved greedy codes to {codes_path}")


if __name__ == "__main__":
    main()
