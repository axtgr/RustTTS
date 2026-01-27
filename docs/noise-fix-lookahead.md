# CodePredictor Look-ahead Implementation Plan

## Problem Statement

Audio noise artifacts ("шорканье") appear at phrase boundaries - specifically at transitions between pause and speech tokens. The noise sounds like scratching/dragging and occurs:
- At the beginning of audio
- At the end of audio  
- Before each phrase/clause after commas (on phrase boundaries)

### Root Cause

The problem is in **CodePredictor** architecture. Current flow:

```
For each acoustic token:
1. Talker generates zeroth_token (semantic codebook 0)
2. Talker produces hidden_states from forward pass
3. CodePredictor receives:
   - hidden_states (from CURRENT token's forward pass, but still contains "pause" context)
   - zeroth_embed (CURRENT token's embedding - already "speech")
4. CodePredictor predicts residual codebooks 1-15
```

**The mismatch:** At pause→speech transition:
- `hidden_states` still contains "pause" context from accumulated KV cache
- `zeroth_embed` already contains "speech" token embedding
- This context mismatch causes CodePredictor to generate inconsistent residual codes
- Result: Audio decoder produces noise artifacts

## Solution: Two-Phase Generation with Look-ahead

### Architecture Overview

Split generation into two phases:
1. **Phase 1:** Generate ALL zeroth tokens first (semantic codebook 0)
2. **Phase 2:** Predict residual codebooks 1-15 with look-ahead context

This allows CodePredictor to "see" the next token when predicting residuals, smoothing transitions.

### Key Parameters

- **Blending coefficient α:** Fixed at 0.75 (current context weight)
- **Memory strategy:** Sliding window of 2 tokens (current + next)
- **Transition detection:** Apply blending always (simpler, can optimize later)

## Implementation Details

### 1. CodePredictor Changes

**File:** `crates/acoustic-model/src/code_predictor.rs`

Add new method:

```rust
/// Predict residual codebooks with look-ahead context.
/// 
/// When next_hidden and next_zeroth_embed are provided, blends current
/// and next context to smooth transitions between different acoustic states
/// (e.g., pause → speech).
///
/// # Arguments
/// * `hidden_states` - Current token's hidden states [batch, 1, hidden]
/// * `zeroth_codes` - Current zeroth token
/// * `zeroth_embed` - Current zeroth embedding
/// * `next_hidden` - OPTIONAL: Next token's hidden states for look-ahead
/// * `next_zeroth_embed` - OPTIONAL: Next token's zeroth embedding
/// * `sampling_config` - Sampling configuration
///
/// # Blending Formula
/// When look-ahead is available:
/// effective_hidden = α * hidden_states + (1-α) * next_hidden
/// where α = 0.75 (favors current context but incorporates future)
pub fn predict_from_hidden_with_lookahead(
    &self,
    hidden_states: &Tensor,
    zeroth_codes: &Tensor,
    zeroth_embed: &Tensor,
    next_hidden: Option<&Tensor>,
    next_zeroth_embed: Option<&Tensor>,
    sampling_config: SamplingConfig,
) -> Result<Tensor>
```

### 2. Model Changes

**File:** `crates/acoustic-model/src/model.rs`

Refactor `generate_acoustic_multi_codebook_from_embeds` into two phases:

```rust
// PHASE 1: Generate all zeroth tokens
let (all_zeroth_tokens, all_hidden_states) = 
    self.generate_all_zeroth_tokens_phase1(...)?;

// PHASE 2: Predict residuals with look-ahead
let all_frames = self.predict_residuals_with_lookahead_phase2(
    &all_zeroth_tokens,
    &all_hidden_states,
    code_predictor,
    sampling_config,
)?;
```

**New private methods:**

```rust
/// Phase 1: Generate all zeroth tokens first.
/// Returns (zeroth_tokens, hidden_states_for_each_token)
fn generate_all_zeroth_tokens_phase1(
    &self,
    prefill_hidden: &Tensor,
    trailing_text_hidden: Option<&Tensor>,
    tts_pad_embed: &Tensor,
    code_predictor: &CodePredictor,
    first_residual_codes: &[u32],
    first_zeroth_token: u32,
    sampling_config: SamplingConfig,
    max_new_tokens: usize,
    min_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<(Vec<u32>, Vec<Tensor>)>;

/// Phase 2: Predict residuals with look-ahead context.
/// Uses sliding window of 2 tokens for memory efficiency.
fn predict_residuals_with_lookahead_phase2(
    &self,
    zeroth_tokens: &[u32],
    hidden_states: &[Tensor],
    code_predictor: &CodePredictor,
    sampling_config: SamplingConfig,
) -> Result<Vec<Vec<u32>>>;
```

### 3. Phase 2 Implementation Detail

```rust
fn predict_residuals_with_lookahead_phase2(...) -> Result<Vec<Vec<u32>>> {
    let mut all_frames = Vec::with_capacity(zeroth_tokens.len());
    
    for i in 0..zeroth_tokens.len() {
        let current_hidden = &hidden_states[i];
        let current_zeroth = zeroth_tokens[i];
        let current_zeroth_tensor = Tensor::new(&[current_zeroth], &self.device)?.unsqueeze(0)?;
        let current_embed = self.embed_codec(&current_zeroth_tensor)?;
        
        // Look-ahead: get next token's context if available
        let (next_hidden, next_embed) = if i + 1 < zeroth_tokens.len() {
            let next_z = zeroth_tokens[i + 1];
            let next_z_tensor = Tensor::new(&[next_z], &self.device)?.unsqueeze(0)?;
            let next_e = self.embed_codec(&next_z_tensor)?;
            (Some(&hidden_states[i + 1]), Some(next_e))
        } else {
            (None, None)
        };
        
        // Predict residuals with look-ahead
        let residuals = code_predictor.predict_from_hidden_with_lookahead(
            current_hidden,
            &current_zeroth_tensor,
            &current_embed,
            next_hidden,
            next_embed.as_ref(),
            sampling_config.clone(),
        )?;
        
        // Build frame
        let residual_codes: Vec<u32> = residuals.squeeze(0)?.squeeze(0)?.i(1..)?.to_vec1()?;
        let mut frame = vec![current_zeroth];
        frame.extend(&residual_codes);
        all_frames.push(frame);
    }
    
    Ok(all_frames)
}
```

### 4. Blending in CodePredictor

```rust
fn predict_from_hidden_with_lookahead(...) -> Result<Tensor> {
    const LOOKAHEAD_ALPHA: f64 = 0.75;  // Weight for current context
    
    // Apply blending if look-ahead context is available
    let effective_hidden = match (next_hidden, next_zeroth_embed) {
        (Some(next_h), Some(_next_e)) => {
            // Blend: α * current + (1-α) * next
            let current_weighted = (hidden_states * LOOKAHEAD_ALPHA)?;
            let next_weighted = (next_h * (1.0 - LOOKAHEAD_ALPHA))?;
            (current_weighted + next_weighted)?
        }
        _ => hidden_states.clone(),
    };
    
    // Rest of the prediction logic uses effective_hidden instead of hidden_states
    let prefill_embeds = Tensor::cat(&[&effective_hidden, zeroth_embed], 1)?;
    // ... continue with transformer layers and residual prediction
}
```

## Memory Considerations

### Sliding Window Strategy

Instead of storing all hidden_states in memory:

```rust
// During Phase 1, store only last 2 hidden states
struct SlidingHiddenBuffer {
    prev: Option<Tensor>,
    current: Option<Tensor>,
}

// During Phase 2, process immediately as we have current + next
```

For a 20-second audio (~250 frames):
- Full storage: ~250 * 1024 * 4 bytes = ~1MB (acceptable)
- Sliding window: ~2 * 1024 * 4 bytes = ~8KB

**Decision:** Start with full storage for simplicity, optimize later if needed.

## Testing Strategy

1. **Generate test audio** with known problematic text
2. **Compare waveforms** at transition points (before/after fix)
3. **Listen test** for noise artifacts
4. **Measure latency impact** (two-phase should be similar or slightly faster due to batching potential)

### Test Command

```bash
cargo build -p tts-app --release --features metal

./target/release/tts synth \
  --model-dir models/qwen3-tts-0.6b-customvoice \
  --codec-dir models/qwen3-tts-tokenizer \
  --speaker vivian \
  --lang ru \
  --multi-codebook \
  -o /tmp/test_lookahead.wav \
  "Ниже представлен план цикла статей, который расскажет о том, что такое Yttri, почему это хорошее и удобное решение."
```

## Rollback Strategy

Add feature flag for A/B testing:

```rust
// In SamplingConfig or separate config
pub struct GenerationConfig {
    pub enable_lookahead: bool,  // default: true
    pub lookahead_alpha: f64,    // default: 0.75
}
```

If issues arise, disable with `--no-lookahead` flag.

## Timeline

1. CodePredictor changes: ~30 min
2. Model refactoring: ~1 hour
3. Testing: ~30 min
4. Fine-tuning α parameter if needed: ~30 min

Total estimate: ~2.5 hours

## Future Optimizations

1. **Smart transition detection:** Only apply blending at pause↔speech boundaries
2. **Adaptive α:** Vary blending coefficient based on token similarity
3. **Batched residual prediction:** Process multiple frames in parallel in Phase 2
