#!/usr/bin/env python3
"""
Dump CodePredictor intermediate values for debugging Rust implementation.

This script extracts:
1. Hidden states from Talker (past_hidden)
2. Zeroth codebook embeddings (last_id_hidden)
3. CodePredictor inputs (concatenated embeddings)
4. CodePredictor logits at each step
5. Predicted residual codes

Output is saved to JSON for comparison with Rust.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add Qwen3-TTS to path
sys.path.insert(0, "/private/tmp/Qwen3-TTS")

# Config
MODEL_PATH = "models/qwen3-tts-0.6b-customvoice"
TEXT = "Hello world"
SPEAKER = "ryan"
LANG = "english"
OUTPUT_DIR = Path("tests/golden")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")

    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models import Qwen3TTSProcessor

    # Load model
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    processor = Qwen3TTSProcessor.from_pretrained(MODEL_PATH)

    print(f"Model loaded successfully")

    # Get the talker model
    talker = model.talker
    code_predictor = talker.code_predictor

    print(f"\nTalker text_embedding: {talker.model.text_embedding.weight.shape}")
    print(f"Talker codec_embedding: {talker.model.codec_embedding.weight.shape}")
    print(f"CodePredictor num embeddings: {len(code_predictor.model.codec_embedding)}")

    print("\n" + "=" * 60)
    print("STEP 1: Analyze embeddings")
    print("=" * 60)

    # Sample zeroth code from reference
    zeroth_code = torch.tensor([[1995]])

    # Get zeroth embedding from talker's codec_embedding
    zeroth_embed = talker.model.codec_embedding(zeroth_code)
    print(f"\nZeroth code {zeroth_code.item()} embedding shape: {zeroth_embed.shape}")
    print(f"Zeroth embed first 10 values: {zeroth_embed[0, 0, :10].tolist()}")

    # Get CodePredictor's codec_embedding for residual codebook 0
    cp_embed_0 = code_predictor.model.codec_embedding[0](zeroth_code)
    print(
        f"\nCodePredictor embed[0] for code {zeroth_code.item()} shape: {cp_embed_0.shape}"
    )
    print(f"CodePredictor embed[0] first 10 values: {cp_embed_0[0, 0, :10].tolist()}")

    print("\n" + "=" * 60)
    print("STEP 2: Manual CodePredictor forward pass")
    print("=" * 60)

    # Create test inputs
    batch_size = 1
    hidden_size = 1024

    # Use deterministic past_hidden for reproducibility
    torch.manual_seed(42)
    dummy_past_hidden = torch.randn(batch_size, 1, hidden_size)

    # Get last_id_hidden from zeroth code embedding
    last_id_hidden = talker.model.codec_embedding(zeroth_code)

    print(f"dummy_past_hidden shape: {dummy_past_hidden.shape}")
    print(f"last_id_hidden shape: {last_id_hidden.shape}")

    # Concatenate as CodePredictor expects
    cp_input = torch.cat((dummy_past_hidden, last_id_hidden), dim=1)
    print(f"CodePredictor input shape: {cp_input.shape}")
    print(f"CodePredictor input first 10 (pos 0): {cp_input[0, 0, :10].tolist()}")
    print(f"CodePredictor input first 10 (pos 1): {cp_input[0, 1, :10].tolist()}")

    # Call CodePredictor generate
    print("\nCalling CodePredictor.generate()...")
    with torch.no_grad():
        cp_result = code_predictor.generate(
            inputs_embeds=cp_input,
            max_new_tokens=15,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    print(f"CodePredictor result sequences shape: {cp_result.sequences.shape}")
    print(f"CodePredictor result sequences: {cp_result.sequences.tolist()}")

    print("\n" + "=" * 60)
    print("STEP 3: Load and compare with reference")
    print("=" * 60)

    # Load reference codes
    ref_path = OUTPUT_DIR / "reference_codes_greedy.json"
    if ref_path.exists():
        with open(ref_path) as f:
            ref_data = json.load(f)
        ref_codes = ref_data["codes"]
        print(f"Loaded reference codes: {len(ref_codes)} frames")
        print(f"First frame: {ref_codes[0]}")
        print(f"  zeroth={ref_codes[0][0]}")
        print(f"  residuals={ref_codes[0][1:]}")
    else:
        print(f"Reference file not found: {ref_path}")
        ref_codes = None

    print("\n" + "=" * 60)
    print("STEP 4: Dump embedding weights")
    print("=" * 60)

    # Talker codec embedding (for zeroth codebook)
    talker_codec_emb = talker.model.codec_embedding.weight.detach().numpy()
    print(f"Talker codec_embedding shape: {talker_codec_emb.shape}")

    # CodePredictor codec embeddings (for residual codebooks 1-15)
    cp_embedding_shapes = []
    for i, emb in enumerate(code_predictor.model.codec_embedding):
        w = emb.weight.detach().numpy()
        cp_embedding_shapes.append(w.shape)
    print(f"CodePredictor codec_embedding shapes: {cp_embedding_shapes}")

    # Save detailed dump
    dump_data = {
        "text": TEXT,
        "speaker": SPEAKER,
        "language": LANG,
        "reference_frame_0": ref_codes[0] if ref_codes else None,
        "embeddings": {
            "zeroth_code": zeroth_code.item(),
            "zeroth_embed_from_talker_first_20": zeroth_embed[0, 0, :20].tolist(),
            "cp_embed_0_for_1995_first_20": cp_embed_0[0, 0, :20].tolist(),
        },
        "code_predictor_test": {
            "input_shape": list(cp_input.shape),
            "dummy_past_hidden_first_20": dummy_past_hidden[0, 0, :20].tolist(),
            "last_id_hidden_first_20": last_id_hidden[0, 0, :20].tolist(),
            "result_sequences": cp_result.sequences.tolist(),
        },
        "embedding_weights": {
            "talker_codec_row_0_first_20": talker_codec_emb[0, :20].tolist(),
            "talker_codec_row_1995_first_20": talker_codec_emb[1995, :20].tolist(),
        },
    }

    # Add CodePredictor embedding samples
    for i, emb in enumerate(code_predictor.model.codec_embedding):
        w = emb.weight.detach().numpy()
        dump_data["embedding_weights"][f"cp_embedding_{i}_row_0_first_20"] = w[
            0, :20
        ].tolist()

    output_path = OUTPUT_DIR / "code_predictor_debug.json"
    with open(output_path, "w") as f:
        json.dump(dump_data, f, indent=2)
    print(f"\nSaved debug data to {output_path}")

    # Print summary for quick comparison
    print("\n" + "=" * 60)
    print("SUMMARY FOR RUST COMPARISON")
    print("=" * 60)
    print(f"\nTalker codec_embedding[1995] first 10:")
    print(f"  {talker_codec_emb[1995, :10].tolist()}")
    print(f"\nCodePredictor with dummy input produces:")
    print(f"  sequences: {cp_result.sequences.tolist()}")
    if ref_codes:
        print(f"\nPython reference residuals for frame 0:")
        print(f"  {ref_codes[0][1:]}")


if __name__ == "__main__":
    main()
