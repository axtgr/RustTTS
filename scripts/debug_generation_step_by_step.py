#!/usr/bin/env python3
"""
Step-by-step generation debug script.

This script performs manual generation step-by-step to capture:
1. Hidden states at each step (before and after norm)
2. CodePredictor inputs and outputs
3. All intermediate values for comparison with Rust

This bypasses the HuggingFace generate() to have full control.
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
MAX_FRAMES = 10  # Generate only first N frames for debugging


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
    main_config = model.config
    config = model.config.talker_config

    print(
        f"Config: hidden_size={config.hidden_size}, num_code_groups={config.num_code_groups}"
    )
    print(
        f"Vocab: codec_vocab={config.vocab_size}, codec_pad={config.codec_pad_id}, codec_bos={config.codec_bos_id}, codec_eos={config.codec_eos_token_id}"
    )
    print(
        f"TTS tokens: bos={main_config.tts_bos_token_id}, eos={main_config.tts_eos_token_id}, pad={main_config.tts_pad_token_id}"
    )

    # Build input similar to model.generate() but manually
    # Format: <|im_start|>assistant\nHello world<|im_end|>\n<|im_start|>assistant\n
    assistant_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")

    # Get special embeddings (from main config)
    special_tokens = torch.tensor(
        [
            [
                main_config.tts_bos_token_id,
                main_config.tts_eos_token_id,
                main_config.tts_pad_token_id,
            ]
        ]
    )
    special_embeds = talker.text_projection(
        talker.get_text_embeddings()(special_tokens)
    )
    tts_bos_embed, tts_eos_embed, tts_pad_embed = special_embeds.chunk(3, dim=1)
    print(f"tts_pad_embed shape: {tts_pad_embed.shape}")
    print(f"tts_pad_embed first 10: {tts_pad_embed[0, 0, :10].tolist()}")

    # Get speaker embedding
    spk_id = config.spk_id[SPEAKER.lower()]
    speaker_embed = talker.get_input_embeddings()(torch.tensor(spk_id))
    print(f"Speaker ID: {spk_id}")
    print(f"Speaker embed shape: {speaker_embed.shape}")

    # Language ID
    lang_id = config.codec_language_id[LANG.lower()]
    print(f"Language ID: {lang_id}")

    # Build prefill input embeddings (non-streaming mode)
    # Codec prefill: [think_id, think_bos, lang_id, think_eos, speaker, pad, bos]
    codec_prefill_ids = [
        config.codec_think_id,
        config.codec_think_bos_id,
        lang_id,
        config.codec_think_eos_id,
    ]
    codec_prefill_embeds_0 = talker.get_input_embeddings()(
        torch.tensor([codec_prefill_ids])
    )
    codec_prefill_embeds_1 = talker.get_input_embeddings()(
        torch.tensor([[config.codec_pad_id, config.codec_bos_id]])
    )
    codec_prefill_embeds = torch.cat(
        [codec_prefill_embeds_0, speaker_embed.view(1, 1, -1), codec_prefill_embeds_1],
        dim=1,
    )
    print(f"Codec prefill embeds shape: {codec_prefill_embeds.shape}")

    # Role tokens: <|im_start|>assistant\n (first 3 tokens)
    role_embed = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))

    # Combine role with tts_pad + tts_bos + codec
    # tts_pad * N + tts_bos (for positions before codec_bos)
    num_pad_positions = codec_prefill_embeds.shape[1] - 2
    text_part = torch.cat(
        [tts_pad_embed.expand(-1, num_pad_positions, -1), tts_bos_embed], dim=1
    )
    combined_prefill = text_part + codec_prefill_embeds[:, :-1]

    # In non-streaming mode, we prepend all text at once
    # Text tokens: "Hello world" (tokens 3:-5, excluding role and end markers)
    text_tokens = input_ids[:, 3:-5]  # Just the text content
    print(f"Text tokens: {text_tokens.tolist()}")

    text_embed = talker.text_projection(talker.get_text_embeddings()(text_tokens))

    # Combine: role + prefill + (text + tts_eos with pad) + (tts_pad + codec_bos)
    num_text_tokens = text_tokens.shape[1]
    text_with_eos = torch.cat([text_embed, tts_eos_embed], dim=1)
    pad_for_text = talker.get_input_embeddings()(
        torch.tensor([[config.codec_pad_id] * (num_text_tokens + 1)])
    )

    final_part = tts_pad_embed + talker.get_input_embeddings()(
        torch.tensor([[config.codec_bos_id]])
    )

    talker_input_embed = torch.cat(
        [role_embed, combined_prefill, text_with_eos + pad_for_text, final_part], dim=1
    )

    print(f"Talker input embed shape: {talker_input_embed.shape}")

    # trailing_text_hidden for generation steps
    trailing_text_hidden = tts_pad_embed  # In non-streaming mode

    # === PREFILL ===
    print("\n=== PREFILL ===")
    with torch.no_grad():
        # Forward through model (without generate)
        prefill_outputs = talker.model(
            inputs_embeds=talker_input_embed,
            use_cache=True,
            output_hidden_states=True,
        )

        # Hidden states AFTER final norm (this is what goes to codec_head)
        hidden_states = prefill_outputs.last_hidden_state
        print(f"Hidden states shape: {hidden_states.shape}")

        # Get logits
        logits = talker.codec_head(hidden_states)
        print(f"Logits shape: {logits.shape}")

        # Sample first zeroth token (greedy for debugging)
        last_logits = logits[0, -1, :]

        # Get top-5 before suppression
        top_values, top_indices = torch.topk(last_logits, 5)
        print(
            f"Top 5 before suppress: {list(zip(top_indices.tolist(), top_values.tolist()))}"
        )

        # Suppress special tokens (2048+) except EOS
        suppress_mask = torch.zeros_like(last_logits, dtype=torch.bool)
        suppress_mask[2048:] = True
        suppress_mask[config.codec_eos_token_id] = False
        last_logits[suppress_mask] = float("-inf")

        # Greedy sample
        first_zeroth = last_logits.argmax().item()
        print(f"First zeroth token: {first_zeroth}")

        # past_hidden for CodePredictor: hidden states at last position, AFTER norm
        past_hidden = hidden_states[:, -1:, :]
        print(f"past_hidden shape: {past_hidden.shape}")
        print(f"past_hidden first 20: {past_hidden[0, 0, :20].tolist()}")

        # Get embedding for first zeroth token
        zeroth_embed = talker.get_input_embeddings()(torch.tensor([[first_zeroth]]))
        print(f"zeroth_embed shape: {zeroth_embed.shape}")
        print(f"zeroth_embed first 20: {zeroth_embed[0, 0, :20].tolist()}")

        # === CODEPREDICTOR FOR FIRST TOKEN ===
        print("\n=== CODEPREDICTOR FOR FIRST TOKEN ===")

        # inputs_embeds = cat(past_hidden, zeroth_embed)
        cp_inputs_embeds = torch.cat([past_hidden, zeroth_embed], dim=1)
        print(f"CodePredictor inputs_embeds shape: {cp_inputs_embeds.shape}")

        # Call CodePredictor.generate
        cp_result = code_predictor.generate(
            inputs_embeds=cp_inputs_embeds,
            max_new_tokens=config.num_code_groups - 1,  # 15
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        residuals = cp_result.sequences[0].tolist()
        print(f"Residual codes (15): {residuals}")

        # Full first frame
        first_frame = [first_zeroth] + residuals
        print(f"First frame (all 16): {first_frame}")

    # Compare with reference
    ref_path = OUTPUT_DIR / "reference_codes_greedy.json"
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        ref_first_frame = [ref["codes"][0][i] for i in range(16)]
        print(f"\nReference first frame: {ref_first_frame}")
        print(f"Match: {first_frame == ref_first_frame}")

        if first_frame != ref_first_frame:
            print("\n=== DIFFERENCES ===")
            for i in range(16):
                if first_frame[i] != ref_first_frame[i]:
                    print(
                        f"  Codebook {i}: got {first_frame[i]}, expected {ref_first_frame[i]}"
                    )

    # Save debug data
    debug_data = {
        "text": TEXT,
        "input_ids": input_ids.tolist(),
        "prefill_seq_len": talker_input_embed.shape[1],
        "past_hidden_first_20": past_hidden[0, 0, :20].tolist(),
        "past_hidden_shape": list(past_hidden.shape),
        "zeroth_embed_first_20": zeroth_embed[0, 0, :20].tolist(),
        "first_zeroth_token": first_zeroth,
        "residuals": residuals,
        "first_frame": first_frame,
        "tts_pad_embed_first_20": tts_pad_embed[0, 0, :20].tolist(),
    }

    output_path = OUTPUT_DIR / "debug_generation.json"
    with open(output_path, "w") as f:
        json.dump(debug_data, f, indent=2)
    print(f"\nSaved debug data to {output_path}")


if __name__ == "__main__":
    main()
