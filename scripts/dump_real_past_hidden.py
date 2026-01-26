#!/usr/bin/env python3
"""
Dump REAL past_hidden from actual generation for Rust comparison.

We need to capture the actual past_hidden that gets passed to CodePredictor
during real generation, not a dummy one.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/private/tmp/Qwen3-TTS")

MODEL_PATH = "models/qwen3-tts-0.6b-customvoice"
TEXT = "Hello world"
SPEAKER = "ryan"
LANG = "english"
OUTPUT_DIR = Path("tests/golden")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")

    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models import Qwen3TTSProcessor

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    processor = Qwen3TTSProcessor.from_pretrained(MODEL_PATH)
    print(f"Model loaded")

    talker = model.talker
    code_predictor = talker.code_predictor

    # Capture CodePredictor inputs during generation
    captured_inputs = []
    original_cp_generate = code_predictor.generate

    def patched_cp_generate(inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            captured_inputs.append(
                {
                    "shape": list(inputs_embeds.shape),
                    "pos_0_first_20": inputs_embeds[0, 0, :20].tolist(),
                    "pos_1_first_20": inputs_embeds[0, 1, :20].tolist()
                    if inputs_embeds.shape[1] > 1
                    else None,
                }
            )
        result = original_cp_generate(inputs_embeds=inputs_embeds, **kwargs)
        if hasattr(result, "sequences"):
            captured_inputs[-1]["sequences"] = result.sequences.tolist()
        return result

    code_predictor.generate = patched_cp_generate

    # Capture past_hidden from Talker output
    captured_past_hidden = []
    original_talker_forward = talker.forward

    def patched_talker_forward(*args, **kwargs):
        result = original_talker_forward(*args, **kwargs)
        if hasattr(result, "past_hidden") and result.past_hidden is not None:
            captured_past_hidden.append(
                {
                    "shape": list(result.past_hidden.shape),
                    "first_20": result.past_hidden[0, 0, :20].tolist(),
                }
            )
        return result

    talker.forward = patched_talker_forward

    # Build input
    assistant_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs: {input_ids.tolist()}")

    # Generate just 5 frames for debugging
    print("\nGenerating 5 frames...")

    # The issue is model.generate doesn't expose subtalker params properly
    # Let's try to call talker.generate directly with proper setup

    # Actually, let's look at how the reference was generated
    # and extract the intermediate values from saved data

    # For now, just run generation and see what we capture
    try:
        with torch.no_grad():
            # Use the low-level API if possible
            talker_codes_list, hidden_states_list = model.generate(
                input_ids=[input_ids],
                instruct_ids=None,
                languages=[LANG],
                speakers=[SPEAKER],
                non_streaming_mode=True,
                max_new_tokens=5,
                do_sample=False,
            )

        print(f"\nGenerated {len(talker_codes_list)} sequences")
        if talker_codes_list:
            codes = talker_codes_list[0]
            print(f"Codes: {codes.tolist()[:5]}")
    except Exception as e:
        print(f"Generation failed: {e}")
        print("Trying alternative approach...")

    print(f"\nCaptured {len(captured_inputs)} CodePredictor inputs")
    for i, cap in enumerate(captured_inputs[:5]):
        print(f"\n--- CodePredictor call {i} ---")
        print(f"Shape: {cap['shape']}")
        print(f"Pos 0 first 20: {cap['pos_0_first_20']}")
        print(f"Pos 1 first 20: {cap['pos_1_first_20']}")
        if "sequences" in cap:
            print(f"Sequences: {cap['sequences']}")

    print(f"\nCaptured {len(captured_past_hidden)} past_hidden values")
    for i, cap in enumerate(captured_past_hidden[:5]):
        print(f"\n--- past_hidden {i} ---")
        print(f"Shape: {cap['shape']}")
        print(f"First 20: {cap['first_20']}")

    # Save captured data
    dump_data = {
        "text": TEXT,
        "input_ids": input_ids.tolist(),
        "code_predictor_inputs": captured_inputs[:10],
        "past_hidden_values": captured_past_hidden[:10],
    }

    output_path = OUTPUT_DIR / "real_past_hidden.json"
    with open(output_path, "w") as f:
        json.dump(dump_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
