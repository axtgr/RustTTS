# Phase 7: Real Model Integration

## Overview

This document describes the plan for integrating real Qwen3-TTS model weights into our Rust implementation.

## Model Architecture Analysis

Based on HuggingFace model inspection, Qwen3-TTS consists of:

### 1. Main LM (Talker) - `Qwen3-TTS-12Hz-0.6B-Base`

```
Architecture: Qwen3TTSForConditionalGeneration
Parameters:
  - hidden_size: 1024
  - num_hidden_layers: 28
  - num_attention_heads: 16
  - num_key_value_heads: 8 (GQA)
  - intermediate_size: 3072
  - head_dim: 128
  - text_vocab_size: 151936
  - codec_vocab_size: 3072
  - num_code_groups: 16 (multi-codebook)
  - max_position_embeddings: 32768
  - rope_theta: 1000000
  - rms_norm_eps: 1e-6
  - hidden_act: silu
  
Special Tokens:
  - tts_bos_token_id: 151672
  - tts_eos_token_id: 151673
  - tts_pad_token_id: 151671
  - codec_bos_id: 2149
  - codec_eos_token_id: 2150
  - codec_pad_id: 2148
```

### 2. Code Predictor (for delay pattern)

```
Architecture: Qwen3TTSTalkerCodePredictor
Parameters:
  - hidden_size: 1024
  - num_hidden_layers: 5
  - num_attention_heads: 16
  - num_key_value_heads: 8
  - intermediate_size: 3072
  - num_code_groups: 16
  - vocab_size: 2048 (per codebook)
```

### 3. Audio Tokenizer - `Qwen3-TTS-Tokenizer-12Hz`

```
Architecture: Qwen3TTSTokenizerV2Model
Parameters:
  Encoder:
    - sample_rate: 24000
    - num_quantizers: 32 (uses 16 for speech)
    - codebook_size: 2048
    - frame_rate: 12.5 FPS
    - downsample_rate: 1920 (24000/12.5)
    
  Decoder:
    - num_quantizers: 16
    - codebook_size: 2048
    - decoder_dim: 1536
    - num_hidden_layers: 8
    - upsample_rates: [8, 5, 4, 3] = 480
    - upsampling_ratios: [2, 2] = 4
    - total upsample: 1920 samples per token
```

## Files Structure on HuggingFace

### Qwen3-TTS-12Hz-0.6B-Base (2.52 GB)
```
├── config.json              # Model config
├── generation_config.json   # Generation params
├── model.safetensors        # 1.83 GB - main weights
├── vocab.json               # 2.78 MB - text vocab
├── merges.txt               # 1.67 MB - BPE merges
├── tokenizer_config.json    # Tokenizer config
├── preprocessor_config.json 
└── speech_tokenizer/        # Contains audio tokenizer reference
```

### Qwen3-TTS-Tokenizer-12Hz (682 MB)
```
├── config.json              # Tokenizer model config
├── configuration.json       
├── model.safetensors        # 682 MB - encoder/decoder weights
└── preprocessor_config.json
```

## Implementation Plan

### Phase 7.1: Update Configuration

**File:** `crates/acoustic-model/src/config.rs`

1. Add multi-codebook support:
```rust
pub struct AcousticModelConfig {
    // Talker config
    pub hidden_size: usize,           // 1024
    pub num_layers: usize,            // 28
    pub num_attention_heads: usize,   // 16
    pub num_kv_heads: usize,          // 8
    pub intermediate_size: usize,     // 3072
    pub head_dim: usize,              // 128
    
    // Vocab
    pub text_vocab_size: usize,       // 151936
    pub codec_vocab_size: usize,      // 3072 (total)
    pub num_code_groups: usize,       // 16
    pub codebook_size: usize,         // 2048 (per group)
    
    // Position
    pub max_position_embeddings: usize, // 32768
    pub rope_theta: f64,              // 1000000
    pub rms_norm_eps: f64,            // 1e-6
    
    // Special tokens
    pub tts_bos_token_id: u32,        // 151672
    pub tts_eos_token_id: u32,        // 151673
    pub codec_bos_id: u32,            // 2149
    pub codec_eos_id: u32,            // 2150
}
```

### Phase 7.2: Multi-Codebook Architecture

The model uses **16 parallel codebooks** with delay pattern generation.

1. First, Talker generates "delay pattern" tokens
2. Then, CodePredictor expands to all 16 codebooks

```
Input: [text tokens] -> Talker -> [delay pattern tokens]
                                      |
                                      v
                              CodePredictor -> [16 x codebook tokens]
                                      |
                                      v
                              AudioDecoder -> PCM audio
```

### Phase 7.3: Audio Tokenizer Decoder

**File:** `crates/audio-codec-12hz/src/decoder.rs`

The decoder architecture:
1. Embedding lookup for 16 codebooks
2. Transformer layers (8)
3. Upsampling convolutions: [8, 5, 4, 3] x [2, 2]
4. Output: 1920 samples per token frame

### Phase 7.4: Text Tokenizer

Use HuggingFace tokenizers with:
- vocab.json (151936 tokens)
- merges.txt (BPE merges)

Already integrated via `tokenizers` crate.

## Weight Loading

### Safetensors Keys Pattern

**Main model (model.safetensors):**
```
model.embed_tokens.weight           # [151936, 1024]
model.layers.{0-27}.self_attn.q_proj.weight
model.layers.{0-27}.self_attn.k_proj.weight
model.layers.{0-27}.self_attn.v_proj.weight
model.layers.{0-27}.self_attn.o_proj.weight
model.layers.{0-27}.mlp.gate_proj.weight
model.layers.{0-27}.mlp.up_proj.weight
model.layers.{0-27}.mlp.down_proj.weight
model.layers.{0-27}.input_layernorm.weight
model.layers.{0-27}.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

**Code predictor:**
```
code_predictor.layers.{0-4}.*
code_predictor.norm.weight
code_predictor.lm_head.weight
```

**Audio tokenizer:**
```
encoder.*
decoder.*
quantizers.*
```

## Testing Strategy

### Unit Tests
- Config loading from JSON
- Weight tensor shape validation
- Single forward pass

### Integration Tests
- End-to-end: text -> tokens -> audio
- Comparison with Python reference output

### Golden Tests
- Reference audio samples
- WER/MOS metrics

## Estimated Effort

| Task | Effort | Priority |
|------|--------|----------|
| Config update | 2h | High |
| Multi-codebook model | 8h | High |
| CodePredictor | 4h | High |
| Audio decoder rewrite | 8h | High |
| Weight loading | 4h | High |
| Text tokenizer integration | 2h | Medium |
| Testing | 8h | High |
| **Total** | ~36h | |

## Dependencies

- `candle-core` - tensor operations
- `candle-nn` - neural network layers  
- `safetensors` - weight loading
- `tokenizers` - text tokenization
- `hf-hub` - model downloading (optional)

## Next Steps

1. Download model weights locally:
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./models/qwen3-tts-0.6b
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./models/qwen3-tts-tokenizer
```

2. Inspect safetensors keys:
```bash
python -c "from safetensors import safe_open; f=safe_open('model.safetensors','framework=pt'); print(f.keys())"
```

3. Start implementation from config update

## References

- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- [HuggingFace Model Page](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Original Python Implementation](https://github.com/QwenLM/Qwen3-TTS)
