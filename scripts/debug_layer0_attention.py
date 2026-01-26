#!/usr/bin/env python3
"""
Detailed layer 0 attention debugging.

This script captures all intermediate values in layer 0 attention
to compare with Rust implementation step-by-step.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/private/tmp/Qwen3-TTS")

MODEL_PATH = "models/qwen3-tts-0.6b-customvoice"
TEXT = "Hello"
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

    # Build input embeddings
    assistant_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")

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

    # Access layer 0 directly
    layer0 = talker.model.layers[0]
    attn = layer0.self_attn

    attn_config = attn.config
    num_heads = attn_config.num_attention_heads
    num_kv_heads = attn_config.num_key_value_heads
    head_dim = attn_config.head_dim
    hidden_size = attn_config.hidden_size

    print(f"\n=== LAYER 0 ATTENTION CONFIG ===")
    print(f"num_heads: {num_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
    print(f"head_dim: {head_dim}")
    print(f"hidden_size: {hidden_size}")

    # Manually step through layer 0
    with torch.no_grad():
        x = talker_input_embed  # [1, seq_len, hidden_size]

        # Input layernorm
        normed = layer0.input_layernorm(x)

        print(f"\n=== AFTER INPUT LAYERNORM ===")
        print(f"Last pos, first 20: {normed[0, -1, :20].tolist()}")

        # Q, K, V projections
        q = attn.q_proj(normed)
        k = attn.k_proj(normed)
        v = attn.v_proj(normed)

        print(f"\n=== Q PROJECTION ===")
        print(f"Q shape: {q.shape}")
        print(f"Last pos, first 10: {q[0, -1, :10].tolist()}")

        # Reshape for multi-head
        batch_size = x.shape[0]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        print(f"\n=== AFTER RESHAPE ===")
        print(f"Q shape: {q.shape}")  # [batch, num_heads, seq, head_dim]
        print(f"K shape: {k.shape}")  # [batch, num_kv_heads, seq, head_dim]

        # QK-Norm (Qwen3 specific)
        q = attn.q_norm(q)
        k = attn.k_norm(k)

        print(f"\n=== AFTER QK-NORM ===")
        print(f"Q normed, head 0, last pos, first 10: {q[0, 0, -1, :10].tolist()}")
        print(f"K normed, head 0, last pos, first 10: {k[0, 0, -1, :10].tolist()}")

        # RoPE - need to get rotation embeddings
        # Qwen3-TTS uses multimodal RoPE with 3D position_ids: (3, batch_size, seq_len)
        # For text-only TTS, all 3 components are identical
        position_ids_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(
            0
        )  # [1, seq_len]
        position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1)  # [3, 1, seq_len]

        # Get rotary embeddings from model level
        rotary_emb = talker.model.rotary_emb
        cos, sin = rotary_emb(v, position_ids)
        # cos, sin shape: [3, 1, seq_len, head_dim] for multimodal RoPE
        print(f"\n=== ROTARY EMBEDDINGS ===")
        print(f"cos shape: {cos.shape}")
        print(f"sin shape: {sin.shape}")
        print(f"cos section 0, last pos, first 10: {cos[0, 0, -1, :10].tolist()}")
        print(f"sin section 0, last pos, first 10: {sin[0, 0, -1, :10].tolist()}")

        # mrope_section from config: [24, 20, 20]
        # interleaved = True according to config
        mrope_section = [24, 20, 20]
        mrope_interleaved = True

        # Use the actual implementation from the model
        from qwen_tts.core.models.modeling_qwen3_tts import (
            apply_multimodal_rotary_pos_emb as model_apply_mrope,
        )

        q, k = model_apply_mrope(
            q, k, cos, sin, mrope_section, mrope_interleaved=mrope_interleaved
        )

        print(f"\n=== AFTER ROTARY EMBEDDING ===")
        print(f"Q after RoPE, head 0, last pos, first 10: {q[0, 0, -1, :10].tolist()}")
        print(f"K after RoPE, head 0, last pos, first 10: {k[0, 0, -1, :10].tolist()}")

        # GQA - repeat KV heads
        def repeat_kv(hidden_states, n_rep):
            batch, num_kv_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(
                batch, num_kv_heads, n_rep, slen, head_dim
            )
            return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

        n_rep = num_heads // num_kv_heads
        k_expanded = repeat_kv(k, n_rep)
        v_expanded = repeat_kv(v, n_rep)

        print(f"\n=== AFTER GQA EXPANSION ===")
        print(f"K expanded shape: {k_expanded.shape}")
        print(f"V expanded shape: {v_expanded.shape}")

        # Attention weights (before scaling)
        attn_weights_raw = torch.matmul(q, k_expanded.transpose(-2, -1))
        print(f"\n=== ATTENTION WEIGHTS (Q @ K^T, before scale) ===")
        print(f"Shape: {attn_weights_raw.shape}")
        print(
            f"Head 0, last query, first 12: {attn_weights_raw[0, 0, -1, :12].tolist()}"
        )

        # Scale
        scale = 1.0 / (head_dim**0.5)
        attn_weights = attn_weights_raw * scale

        print(f"\n=== ATTENTION WEIGHTS (after scale, scale={scale}) ===")
        print(f"Head 0, last query, first 12: {attn_weights[0, 0, -1, :12].tolist()}")

        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        attn_weights_masked = attn_weights + causal_mask

        print(f"\n=== ATTENTION WEIGHTS (after causal mask) ===")
        print(
            f"Head 0, last query, first 12: {attn_weights_masked[0, 0, -1, :12].tolist()}"
        )

        # Softmax
        attn_probs = F.softmax(attn_weights_masked, dim=-1, dtype=torch.float32)

        print(f"\n=== ATTENTION PROBS (after softmax) ===")
        print(
            f"Head 0, last query, all {seq_len} probs: {attn_probs[0, 0, -1, :].tolist()}"
        )
        print(f"Sum of probs: {attn_probs[0, 0, -1, :].sum().item()}")

        # Attention output (before o_proj)
        attn_output = torch.matmul(attn_probs, v_expanded)

        print(f"\n=== ATTENTION OUTPUT (before o_proj) ===")
        print(f"Shape: {attn_output.shape}")
        print(f"Head 0, last pos, first 10: {attn_output[0, 0, -1, :10].tolist()}")

        # Reshape for o_proj
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        print(f"\n=== ATTENTION OUTPUT (reshaped for o_proj) ===")
        print(f"Shape: {attn_output.shape}")
        print(f"Last pos, first 20: {attn_output[0, -1, :20].tolist()}")

        # O projection
        attn_out = attn.o_proj(attn_output)

        print(f"\n=== AFTER O_PROJ ===")
        print(f"Last pos, first 20: {attn_out[0, -1, :20].tolist()}")

        # Residual
        hidden_after_attn = x + attn_out

        print(f"\n=== AFTER ATTENTION + RESIDUAL ===")
        print(f"Last pos, first 20: {hidden_after_attn[0, -1, :20].tolist()}")

        # Post attention layernorm
        normed2 = layer0.post_attention_layernorm(hidden_after_attn)

        print(f"\n=== AFTER POST_ATTENTION_LAYERNORM ===")
        print(f"Last pos, first 20: {normed2[0, -1, :20].tolist()}")

        # MLP
        mlp_out = layer0.mlp(normed2)

        print(f"\n=== MLP OUTPUT ===")
        print(f"Last pos, first 20: {mlp_out[0, -1, :20].tolist()}")

        # Final output
        layer_output = hidden_after_attn + mlp_out

        print(f"\n=== LAYER 0 FINAL OUTPUT ===")
        print(f"Last pos, first 20: {layer_output[0, -1, :20].tolist()}")

    # Collect all data for comparison
    debug_data = {
        "text": TEXT,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "scale": scale,
        # Input embed
        "input_embed_last_pos": talker_input_embed[0, -1, :].tolist(),
        # After layernorm
        "after_layernorm_last_pos": normed[0, -1, :].tolist(),
        # Q proj (before reshape)
        "q_proj_last_pos": attn.q_proj(normed)[0, -1, :].tolist(),
        # K proj (before reshape)
        "k_proj_last_pos": attn.k_proj(normed)[0, -1, :].tolist(),
        # Cos/sin for RoPE (raw, before mrope combination)
        "cos_last_pos": cos[0, 0, -1, :].tolist(),
        "sin_last_pos": sin[0, 0, -1, :].tolist(),
        # Q/K after RoPE
        "q_rope_head0_last_pos": q[0, 0, -1, :].tolist(),
        "k_rope_head0_last_pos": k[0, 0, -1, :].tolist(),
        # Q/K normed (before RoPE)
        "q_normed_head0_last_pos": attn.q_norm(
            attn.q_proj(normed)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )[0, 0, -1, :].tolist(),
        "k_normed_head0_last_pos": attn.k_norm(
            attn.k_proj(normed)
            .view(batch_size, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )[0, 0, -1, :].tolist(),
        # Attention weights
        "attn_weights_head0_last_query": attn_weights[0, 0, -1, :].tolist(),
        "attn_probs_head0_last_query": attn_probs[0, 0, -1, :].tolist(),
        # After o_proj
        "after_o_proj_last_pos": attn_out[0, -1, :].tolist(),
        # After attention + residual
        "after_attn_residual_last_pos": hidden_after_attn[0, -1, :].tolist(),
        # MLP output
        "mlp_output_last_pos": mlp_out[0, -1, :].tolist(),
        # Layer 0 final output
        "layer0_output_last_pos": layer_output[0, -1, :].tolist(),
    }

    output_path = OUTPUT_DIR / "layer0_attention_debug.json"
    with open(output_path, "w") as f:
        json.dump(debug_data, f, indent=2)
    print(f"\n\nSaved detailed debug data to {output_path}")


if __name__ == "__main__":
    main()
