#!/usr/bin/env python3
"""
Layer-by-layer hidden states comparison.

This script captures hidden states after each transformer layer
to pinpoint where divergence between Python and Rust begins.
"""

import json
import sys
from pathlib import Path

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
    main_config = model.config
    config = model.config.talker_config

    # Build input embeddings (same as debug_generation_step_by_step.py)
    assistant_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")

    # Get special embeddings
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

    # Get speaker embedding
    spk_id = config.spk_id[SPEAKER.lower()]
    speaker_embed = talker.get_input_embeddings()(torch.tensor(spk_id))

    # Language ID
    lang_id = config.codec_language_id[LANG.lower()]

    # Build prefill embeddings
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

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))

    num_pad_positions = codec_prefill_embeds.shape[1] - 2
    text_part = torch.cat(
        [tts_pad_embed.expand(-1, num_pad_positions, -1), tts_bos_embed], dim=1
    )
    combined_prefill = text_part + codec_prefill_embeds[:, :-1]

    text_tokens = input_ids[:, 3:-5]
    text_embed = talker.text_projection(talker.get_text_embeddings()(text_tokens))

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
    seq_len = talker_input_embed.shape[1]

    # Debug: print input embeddings for first and last position
    print(f"\n=== INPUT EMBEDDINGS ===")
    print(f"Position 0, first 20: {talker_input_embed[0, 0, :20].tolist()}")
    print(
        f"Position {seq_len - 1}, first 20: {talker_input_embed[0, seq_len - 1, :20].tolist()}"
    )

    # Now do forward pass with hook to capture all layer outputs
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (hidden_states, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            # Capture last position hidden state
            layer_outputs[layer_idx] = hidden[0, -1, :20].detach().clone().tolist()

        return hook

    # Register hooks on all decoder layers
    hooks = []
    for i, layer in enumerate(talker.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Also capture after embed_tokens (input to first layer)
    input_to_layers = {}

    def input_hook(module, input, output):
        # This captures the output of attention input (which is input to layer)
        pass

    print(f"\n=== LAYER-BY-LAYER FORWARD ===")
    with torch.no_grad():
        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            use_cache=False,
            output_hidden_states=True,  # This gives us all hidden states!
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # outputs.hidden_states contains hidden states after each layer
    # hidden_states[0] = input embeddings
    # hidden_states[i] = output of layer i-1 (for i > 0)
    # hidden_states[-1] = output of last layer (before final norm)

    all_hidden_states = outputs.hidden_states
    print(f"Number of hidden states: {len(all_hidden_states)}")

    # Collect layer-by-layer data
    layer_data = {
        "text": TEXT,
        "seq_len": seq_len,
        "num_layers": len(all_hidden_states) - 1,
        "layers": {},
    }

    # Input embeddings (position 0)
    layer_data["input_embed_pos0_first20"] = talker_input_embed[0, 0, :20].tolist()
    layer_data["input_embed_last_pos_first20"] = talker_input_embed[0, -1, :20].tolist()

    for i, hs in enumerate(all_hidden_states):
        layer_name = f"layer_{i}" if i > 0 else "input"
        # Last position, first 20 values
        vals = hs[0, -1, :20].tolist()
        layer_data["layers"][layer_name] = vals

        if i == 0:
            print(f"Input (before layers), last pos, first 20: {vals[:10]}...")
        elif i <= 5 or i >= len(all_hidden_states) - 3:
            print(f"After layer {i - 1}, last pos, first 20: {vals[:10]}...")

    # Final hidden state after norm
    final_hidden = outputs.last_hidden_state
    layer_data["final_after_norm_first20"] = final_hidden[0, -1, :20].tolist()
    print(
        f"\nFinal (after norm), last pos, first 20: {final_hidden[0, -1, :10].tolist()}..."
    )

    # Also check what norm does
    # The model.norm is applied to get last_hidden_state from hidden_states[-1]
    pre_norm = all_hidden_states[-1]
    layer_data["pre_norm_first20"] = pre_norm[0, -1, :20].tolist()
    print(f"Pre-norm (last layer output), first 20: {pre_norm[0, -1, :10].tolist()}...")

    # Manually apply norm to verify
    with torch.no_grad():
        manual_normed = talker.model.norm(pre_norm)
    layer_data["manual_norm_first20"] = manual_normed[0, -1, :20].tolist()
    print(f"Manual norm result, first 20: {manual_normed[0, -1, :10].tolist()}...")

    # Save data
    output_path = OUTPUT_DIR / "layer_by_layer.json"
    with open(output_path, "w") as f:
        json.dump(layer_data, f, indent=2)
    print(f"\nSaved layer-by-layer data to {output_path}")

    # Also save input embedding stats for verification
    embed_stats = {
        "input_embed_mean": talker_input_embed.mean().item(),
        "input_embed_std": talker_input_embed.std().item(),
        "input_embed_min": talker_input_embed.min().item(),
        "input_embed_max": talker_input_embed.max().item(),
    }
    print(f"\nInput embed stats: {embed_stats}")


if __name__ == "__main__":
    main()
