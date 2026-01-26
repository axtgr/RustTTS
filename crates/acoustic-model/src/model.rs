//! Main acoustic model implementation.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};
use tracing::{debug, info, instrument, warn};

use crate::code_predictor::CodePredictor;
use crate::config::AcousticModelConfig;
use crate::layers::{RmsNorm, RotaryEmbedding, TextProjection, TransformerBlock};
use crate::sampling::{Sampler, SamplingConfig, apply_repetition_penalty};

/// KV cache type alias for transformer layers.
pub type KvCache = Vec<(Tensor, Tensor)>;

/// Forward output with hidden states: (logits, hidden_states, kv_caches).
pub type ForwardWithHiddenOutput = (Tensor, Tensor, KvCache);

/// The main acoustic model for Qwen3-TTS (Talker).
///
/// Qwen3-TTS architecture has:
/// - `text_embedding`: [vocab_size, embedding_dim] - text token embeddings
/// - `text_projection`: FC(embedding_dim→embedding_dim) + SiLU + FC(embedding_dim→hidden_size)
/// - `codec_embedding`: [codec_vocab_size, hidden_size] - codec token embeddings
/// - Transformer layers with QK-Norm
/// - `codec_head`: [codec_vocab_size, hidden_size] - output projection
#[derive(Debug)]
pub struct Model {
    /// Text token embedding layer [vocab_size, embedding_dim].
    text_embedding: Embedding,
    /// Text projection from embedding_dim to hidden_size.
    text_projection: Option<TextProjection>,
    /// Codec token embedding layer [codec_vocab_size, hidden_size].
    codec_embedding: Embedding,
    /// Transformer decoder blocks.
    layers: Vec<TransformerBlock>,
    /// Final layer normalization.
    norm: RmsNorm,
    /// Output projection to codec vocabulary.
    codec_head: candle_nn::Linear,
    /// Rotary position embeddings.
    rotary_emb: RotaryEmbedding,
    /// Model configuration.
    config: AcousticModelConfig,
    /// Device (CPU or CUDA).
    device: Device,
}

impl Model {
    /// Load model from a pretrained directory.
    ///
    /// Expects the directory to contain:
    /// - `config.json` - model configuration
    /// - `model.safetensors` - model weights
    #[instrument(skip_all, fields(path = %dir.as_ref().display()))]
    pub fn from_pretrained(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        info!("Loading model from pretrained directory: {}", dir.display());

        // Load config
        let config = AcousticModelConfig::from_pretrained(dir).map_err(candle_core::Error::msg)?;

        // Load weights
        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(candle_core::Error::msg(format!(
                "model.safetensors not found in {}",
                dir.display()
            )));
        }

        Self::load(&weights_path, config, device)
    }

    /// Load model from safetensors file.
    #[instrument(skip(config), fields(path = %path.as_ref().display()))]
    pub fn load(
        path: impl AsRef<Path>,
        config: AcousticModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading model from {}", path.display());

        // Load tensors from safetensors file
        let tensors = candle_core::safetensors::load(path, device)?;

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

        Self::from_vb(vb, config, device)
    }

    /// Create model from VarBuilder (for testing or custom loading).
    pub fn from_vb(vb: VarBuilder, config: AcousticModelConfig, device: &Device) -> Result<Self> {
        // Qwen3-TTS format: talker.model.*, talker.text_projection.*, talker.codec_head.*
        let vb_talker = vb.pp("talker");
        let vb_model = vb_talker.pp("model");

        // Text embedding layer [vocab_size, embedding_dim]
        // Qwen3-TTS: talker.model.text_embedding.weight [151936, 2048]
        let text_embedding = embedding(
            config.text_vocab_size,
            config.embedding_dim,
            vb_model.pp("text_embedding"),
        )
        .or_else(|e| {
            warn!("Failed to load text_embedding from talker.model: {}", e);
            // Fallback to standard format
            embedding(
                config.text_vocab_size,
                config.embedding_dim,
                vb.pp("model.embed_tokens"),
            )
        })?;

        // Text projection: embedding_dim -> hidden_size
        // Only needed if embedding_dim != hidden_size
        let text_projection = if config.embedding_dim != config.hidden_size {
            match TextProjection::new(
                config.embedding_dim,
                config.hidden_size,
                vb_talker.pp("text_projection"),
            ) {
                Ok(proj) => {
                    info!(
                        "Loaded text_projection: {} -> {}",
                        config.embedding_dim, config.hidden_size
                    );
                    Some(proj)
                }
                Err(e) => {
                    warn!("Failed to load text_projection: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Codec embedding layer [codec_vocab_size, hidden_size]
        // Qwen3-TTS: talker.model.codec_embedding.weight [3072, 1024]
        let codec_embedding = embedding(
            config.codec_vocab_size,
            config.hidden_size,
            vb_model.pp("codec_embedding"),
        )?;

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            debug!("Loading layer {}/{}", i + 1, config.num_layers);
            let layer = TransformerBlock::new(&config, vb_model.pp(format!("layers.{i}")))?;
            layers.push(layer);
        }

        // Final norm
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb_model.pp("norm"))?;

        // Codec head (outputs to codec vocabulary)
        // Qwen3-TTS: talker.codec_head.weight [3072, 1024]
        let codec_head = candle_nn::linear_no_bias(
            config.hidden_size,
            config.codec_vocab_size,
            vb_talker.pp("codec_head"),
        )
        .or_else(|e| {
            warn!("Failed to load codec_head from talker: {}", e);
            // Fallback to lm_head
            candle_nn::linear(
                config.hidden_size,
                config.codec_vocab_size,
                vb.pp("lm_head"),
            )
        })?;

        // Rotary embeddings
        let head_dim = config.head_dim;
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            device,
        )?;

        info!(
            "Model loaded: {} layers, hidden={}, embedding={}, text_vocab={}, codec_vocab={}",
            config.num_layers,
            config.hidden_size,
            config.embedding_dim,
            config.text_vocab_size,
            config.codec_vocab_size
        );

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            rotary_emb,
            config,
            device: device.clone(),
        })
    }

    /// Get the model configuration.
    pub fn config(&self) -> &AcousticModelConfig {
        &self.config
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Embed text tokens (apply text_embedding + text_projection).
    fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embedded = self.text_embedding.forward(input_ids)?;
        if let Some(ref proj) = self.text_projection {
            proj.forward(&embedded)
        } else {
            Ok(embedded)
        }
    }

    /// Embed codec tokens.
    fn embed_codec(&self, codec_ids: &Tensor) -> Result<Tensor> {
        self.codec_embedding.forward(codec_ids)
    }

    /// Forward pass through the model.
    ///
    /// `input_ids` can be either text tokens or codec tokens.
    /// For text tokens (< text_vocab_size), applies text_embedding + text_projection.
    /// For codec tokens, applies codec_embedding.
    ///
    /// Returns logits for the next token and updated KV caches.
    #[instrument(skip(self, input_ids, kv_caches))]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[(Tensor, Tensor)]>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        // For now, assume all input_ids are text tokens or codec tokens uniformly
        // In practice, need to handle mixed sequences based on token ranges
        // TODO: Handle mixed text/codec sequences properly

        // Embed input tokens - use text embedding for now
        let mut hidden_states = self.embed_text(input_ids)?;

        // Pass through transformer layers
        let mut new_kv_caches = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let kv_cache = kv_caches.map(|caches| (&caches[i].0, &caches[i].1));

            let (new_hidden, k_cache, v_cache) =
                layer.forward(&hidden_states, &self.rotary_emb, position_offset, kv_cache)?;

            hidden_states = new_hidden;
            new_kv_caches.push((k_cache, v_cache));
        }

        // Final norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // Project to codec vocabulary
        let logits = self.codec_head.forward(&hidden_states)?;

        Ok((logits, new_kv_caches))
    }

    // Note: generate() and generate_with_hidden() methods have been removed.
    // Use generate_from_embeds() or generate_from_embeds_with_predictor() instead.
    // These methods use the proper dual-track embedding format for Qwen3-TTS.

    /// Generate acoustic tokens from combined embeddings (Qwen3-TTS CustomVoice format).
    ///
    /// In Qwen3-TTS, the input to the talker is a **sum** of text_embedding and codec_embedding,
    /// not a concatenation. This method generates from pre-computed combined embeddings.
    ///
    /// **Important:** During generation, each generated codec token must be combined with
    /// `tts_pad` embedding to maintain the dual-track pattern expected by the model.
    ///
    /// # Arguments
    /// - `combined_embeds`: Pre-computed embeddings [1, seq_len, hidden_size] = text_embed + codec_embed
    /// - `tts_pad_token_id`: The TTS pad token ID to combine with generated codec tokens
    /// - `max_new_tokens`: Maximum number of tokens to generate
    /// - `sampling_config`: Sampling configuration
    /// - `eos_token_id`: EOS token ID to stop generation
    /// - `min_new_tokens`: Minimum tokens before EOS is allowed (prevents early termination)
    ///
    /// # Returns
    /// Tuple of (tokens, hidden_states)
    #[instrument(skip(self, combined_embeds, sampling_config), fields(max_new_tokens))]
    pub fn generate_from_embeds(
        &self,
        combined_embeds: &Tensor,
        tts_pad_token_id: u32,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
        min_new_tokens: usize,
    ) -> Result<(Vec<u32>, Tensor)> {
        let seq_len = combined_embeds.dim(1)?;
        let repetition_penalty = sampling_config.repetition_penalty;
        info!(
            "Generating up to {} tokens (min={}) from {} prefill embeddings, tts_pad={}, rep_penalty={}",
            max_new_tokens, min_new_tokens, seq_len, tts_pad_token_id, repetition_penalty
        );

        let mut sampler = Sampler::new(sampling_config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut all_hidden_states: Vec<Tensor> = Vec::with_capacity(max_new_tokens);

        // Suppress tokens: special tokens (2048-3071) except EOS
        // Audio tokens are 0-2047, special tokens start at 2048
        let suppress_start = 2048u32;
        let suppress_end = self.config.codec_vocab_size as u32;

        // Pre-compute tts_pad embedding for dual-track generation
        // During generation, each codec token is combined with tts_pad embedding
        let tts_pad_tensor = Tensor::new(&[tts_pad_token_id], &self.device)?.unsqueeze(0)?;
        let tts_pad_embed = self.embed_text(&tts_pad_tensor)?; // [1, 1, hidden_size]

        // Step 1: Prefill with combined embeddings
        let (logits, hidden_states, kv_caches) = self.forward_embeds(combined_embeds, 0, None)?;

        let mut position_offset = seq_len;
        let mut kv_caches: Option<Vec<(Tensor, Tensor)>> = Some(kv_caches);

        // Sample first codec token from last position's logits
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
        let mut logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

        // Log top logits before suppression for debugging
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
        info!("Top 5 logits before suppress (step 0): {:?}", top_5);

        // Suppress special tokens (except EOS)
        Self::suppress_special_tokens(&mut logits_vec, suppress_start, suppress_end, eos_token_id);

        // Log top logits after suppression
        let mut indexed_after: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        indexed_after.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_5_after: Vec<(usize, f32)> = indexed_after.iter().take(5).copied().collect();
        info!("Top 5 logits after suppress (step 0): {:?}", top_5_after);

        let mut current_token = sampler.sample(&logits_vec);

        debug!("Step 0: sampled token {} from prefill", current_token);

        // Check for EOS
        if Some(current_token) == eos_token_id {
            info!("EOS token generated at first step");
            let empty_hidden =
                Tensor::zeros((1, 0, self.config.hidden_size), DType::F32, &self.device)?;
            return Ok((generated, empty_hidden));
        }
        generated.push(current_token);
        // Store hidden state from prefill for the first token
        // This is the hidden state at the last position of prefill, which produced current_token
        all_hidden_states.push(hidden_states.i((.., seq_len - 1..seq_len, ..))?.clone());

        // Generate remaining tokens with dual-track: tts_pad + codec token
        for step in 1..max_new_tokens {
            // Embed current codec token
            let codec_tensor = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;
            let codec_embed = self.embed_codec(&codec_tensor)?;

            // DUAL-TRACK: Combine codec embedding with tts_pad embedding
            // This maintains the pattern expected by the model during generation
            let combined_step_embed = (&tts_pad_embed + &codec_embed)?;

            // Forward pass with combined dual-track embedding
            let (logits, hidden_states, new_kv_caches) =
                self.forward_embeds(&combined_step_embed, position_offset, kv_caches.as_deref())?;

            // Get logits for the last position
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let mut logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

            // Suppress special tokens (except EOS when allowed)
            // Also suppress EOS if we haven't reached min_new_tokens yet
            let effective_eos = if generated.len() < min_new_tokens {
                None // Don't allow EOS yet - suppress all 2048+
            } else {
                eos_token_id
            };
            Self::suppress_special_tokens(
                &mut logits_vec,
                suppress_start,
                suppress_end,
                effective_eos,
            );

            // Apply repetition penalty to discourage repeated tokens
            apply_repetition_penalty(&mut logits_vec, &generated, repetition_penalty);

            // Log top logits for early steps (after suppression and repetition penalty)
            if step <= 3 {
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
                info!(
                    "Top 5 logits after suppress+rep_penalty (step {}): {:?}",
                    step, top_5
                );
            }

            // Sample next token
            let next_token = sampler.sample(&logits_vec);

            // Log every 50 steps or when close to EOS
            if step % 50 == 0 || next_token >= 2048 {
                info!(
                    "Step {}: token={}, eos_token={:?}, is_eos={}, generated={}",
                    step,
                    next_token,
                    eos_token_id,
                    Some(next_token) == eos_token_id,
                    generated.len()
                );
            }
            debug!("Step {}: generated token {}", step, next_token);

            // Check for EOS (only after min_new_tokens)
            if Some(next_token) == eos_token_id && generated.len() >= min_new_tokens {
                info!(
                    "EOS token generated at step {} (after {} tokens)",
                    step,
                    generated.len()
                );
                break;
            }

            // Store hidden state for current_token (the token we just processed)
            // Only store if we're continuing generation (not EOS)
            all_hidden_states.push(hidden_states.clone());

            generated.push(next_token);

            // Update for next iteration
            kv_caches = Some(new_kv_caches);
            position_offset += 1;
            current_token = next_token;
        }

        info!("Generated {} tokens", generated.len());

        // Concatenate all hidden states: [1, num_tokens, hidden_size]
        let concatenated_hidden = if all_hidden_states.is_empty() {
            Tensor::zeros((1, 0, self.config.hidden_size), DType::F32, &self.device)?
        } else {
            Tensor::cat(&all_hidden_states, 1)?
        };

        Ok((generated, concatenated_hidden))
    }

    /// Generate acoustic tokens from combined embeddings with CodePredictor integration.
    ///
    /// This method implements the correct Qwen3-TTS generation flow where each step
    /// uses the **sum of all 16 codebook embeddings** from the previous step:
    ///
    /// ```text
    /// Python SDK logic (modeling_qwen3_tts.py:1689-1692):
    /// codec_hiddens = cat([zeroth_embed] + [code_predictor_embed[i](codes[i]) for i in 0..15])
    /// inputs_embeds = codec_hiddens.sum(dim=1)
    /// if generation_step < trailing_text_hidden.shape[1]:
    ///     inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step]
    /// else:
    ///     inputs_embeds = inputs_embeds + tts_pad_embed
    /// ```
    ///
    /// This is critical for generating tokens that match Python SDK output.
    ///
    /// # Arguments
    /// - `combined_embeds`: Pre-computed embeddings [1, seq_len, hidden_size] = text_embed + codec_embed
    /// - `tts_pad_token_id`: The TTS pad token ID to use after text_hidden is exhausted
    /// - `code_predictor`: CodePredictor for predicting residual codebooks
    /// - `max_new_tokens`: Maximum number of tokens to generate
    /// - `sampling_config`: Sampling configuration
    /// - `eos_token_id`: EOS token ID to stop generation
    /// - `min_new_tokens`: Minimum tokens before EOS is allowed
    /// - `trailing_text_hidden`: Optional text embeddings [1, num_text_tokens, hidden_size] for conditioning
    ///   If provided, used for first N steps; after that tts_pad_embed is used.
    ///
    /// # Returns
    /// Tuple of (zeroth_tokens, all_tokens_interleaved, hidden_states)
    #[allow(clippy::too_many_arguments)]
    #[instrument(
        skip(
            self,
            combined_embeds,
            code_predictor,
            sampling_config,
            trailing_text_hidden
        ),
        fields(max_new_tokens)
    )]
    pub fn generate_from_embeds_with_predictor(
        &self,
        combined_embeds: &Tensor,
        tts_pad_token_id: u32,
        code_predictor: &CodePredictor,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
        min_new_tokens: usize,
        trailing_text_hidden: Option<&Tensor>,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>, Tensor)> {
        let seq_len = combined_embeds.dim(1)?;
        let repetition_penalty = sampling_config.repetition_penalty;
        info!(
            "Generating up to {} tokens (min={}) from {} prefill embeddings with CodePredictor, tts_pad={}, rep_penalty={}",
            max_new_tokens, min_new_tokens, seq_len, tts_pad_token_id, repetition_penalty
        );

        let mut sampler = Sampler::new(sampling_config.clone());
        let mut generated_zeroth: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut all_frames: Vec<Vec<u32>> = Vec::with_capacity(max_new_tokens);
        let mut all_hidden_states: Vec<Tensor> = Vec::with_capacity(max_new_tokens);

        // Suppress tokens: special tokens (2048-3071) except EOS
        let suppress_start = 2048u32;
        let suppress_end = self.config.codec_vocab_size as u32;

        // Pre-compute tts_pad embedding
        let tts_pad_tensor = Tensor::new(&[tts_pad_token_id], &self.device)?.unsqueeze(0)?;
        let tts_pad_embed = self.embed_text(&tts_pad_tensor)?; // [1, 1, hidden_size]

        // Get trailing_text length for conditioning
        let trailing_text_len = trailing_text_hidden
            .map(|t| t.dim(1).unwrap_or(0))
            .unwrap_or(0);
        info!(
            "Trailing text hidden: {:?}, will use for first {} steps",
            trailing_text_hidden.map(|t| t.dims().to_vec()),
            trailing_text_len
        );

        // Step 1: Prefill with combined embeddings
        let (logits, hidden_states, kv_caches) = self.forward_embeds(combined_embeds, 0, None)?;

        let mut position_offset = seq_len;
        let mut kv_caches: Option<Vec<(Tensor, Tensor)>> = Some(kv_caches);

        // Sample first zeroth token from last position's logits
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
        let mut logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

        // Log top logits before suppression
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
        info!("Top 5 logits before suppress (step 0): {:?}", top_5);

        Self::suppress_special_tokens(&mut logits_vec, suppress_start, suppress_end, eos_token_id);

        let mut current_zeroth_token = sampler.sample(&logits_vec);

        debug!(
            "Step 0: sampled zeroth token {} from prefill",
            current_zeroth_token
        );

        // Check for EOS
        if Some(current_zeroth_token) == eos_token_id {
            info!("EOS token generated at first step");
            let empty_hidden =
                Tensor::zeros((1, 0, self.config.hidden_size), DType::F32, &self.device)?;
            return Ok((generated_zeroth, all_frames, empty_hidden));
        }

        generated_zeroth.push(current_zeroth_token);
        all_hidden_states.push(hidden_states.i((.., seq_len - 1..seq_len, ..))?.clone());

        // Predict residual codebooks for first token using CodePredictor
        let first_hidden = hidden_states.i((.., seq_len - 1..seq_len, ..))?;
        let zeroth_tensor = Tensor::new(&[current_zeroth_token], &self.device)?.unsqueeze(0)?;
        let zeroth_embed = self.embed_codec(&zeroth_tensor)?; // [1, 1, hidden_size]
        let first_residuals = code_predictor.predict_from_hidden(
            &first_hidden,
            &zeroth_tensor,
            &zeroth_embed,
            sampling_config.clone(),
        )?;
        // first_residuals shape: [1, 1, 16] - extract residuals (codebooks 1-15)
        let first_residual_codes: Vec<u32> =
            first_residuals.squeeze(0)?.squeeze(0)?.i(1..)?.to_vec1()?;

        // Store first frame (all 16 codebooks)
        let mut first_frame = vec![current_zeroth_token];
        first_frame.extend(&first_residual_codes);
        all_frames.push(first_frame);

        info!(
            "Step 0: zeroth={}, residuals={:?}",
            current_zeroth_token,
            &first_residual_codes[..3.min(first_residual_codes.len())]
        );

        // For next step, we need: sum of all 16 embeddings + tts_pad
        // zeroth embedding comes from self.codec_embedding
        // residual embeddings come from code_predictor.codec_embeddings[i]
        let mut prev_residual_codes = first_residual_codes;

        // Generate remaining tokens
        for step in 1..max_new_tokens {
            // Build combined embedding for this step:
            // inputs_embeds = zeroth_embed + sum(residual_embeds[i]) + text_conditioning
            // where text_conditioning = trailing_text_hidden[step] if step < len else tts_pad_embed

            // 1. Zeroth codebook embedding (from main model)
            let zeroth_tensor = Tensor::new(&[current_zeroth_token], &self.device)?.unsqueeze(0)?;
            let zeroth_embed = self.embed_codec(&zeroth_tensor)?; // [1, 1, hidden]

            // 2. Sum of residual embeddings (from CodePredictor)
            let residual_tensor =
                Tensor::new(prev_residual_codes.as_slice(), &self.device)?.unsqueeze(0)?;
            let residual_sum = code_predictor.sum_residual_embeddings(&residual_tensor)?; // [1, 1, hidden]

            // 3. Get text conditioning: trailing_text_hidden[generation_step] or tts_pad_embed
            // Python: if generation_step < trailing_text_hidden.shape[1]:
            //             inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step]
            //         else:
            //             inputs_embeds = inputs_embeds + tts_pad_embed
            //
            // Note: Python's generation_step is 0-indexed starting from the first generated token.
            // Our `step` starts at 1 in the loop (after prefill generates token 0).
            // So we need to use `step - 1` as the index into trailing_text_hidden.
            let generation_step = step - 1; // 0-indexed generation step
            let text_conditioning = if generation_step < trailing_text_len {
                // Use trailing text hidden for this generation step
                trailing_text_hidden
                    .unwrap()
                    .narrow(1, generation_step, 1)? // [1, 1, hidden]
            } else {
                // Text exhausted, use tts_pad
                tts_pad_embed.clone()
            };

            // 4. Combine: zeroth + residual_sum + text_conditioning
            let combined_step_embed = ((&zeroth_embed + &residual_sum)? + &text_conditioning)?;

            // Forward pass
            let (logits, hidden_states, new_kv_caches) =
                self.forward_embeds(&combined_step_embed, position_offset, kv_caches.as_deref())?;

            // Get logits for the last position
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let mut logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

            // Suppress special tokens
            let effective_eos = if generated_zeroth.len() < min_new_tokens {
                None
            } else {
                eos_token_id
            };
            Self::suppress_special_tokens(
                &mut logits_vec,
                suppress_start,
                suppress_end,
                effective_eos,
            );

            // Apply repetition penalty
            apply_repetition_penalty(&mut logits_vec, &generated_zeroth, repetition_penalty);

            // Log top logits for early steps
            if step <= 3 {
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
                info!(
                    "Top 5 logits after suppress+rep_penalty (step {}): {:?}",
                    step, top_5
                );
            }

            // Sample next zeroth token
            let next_zeroth = sampler.sample(&logits_vec);

            // Log progress
            if step % 50 == 0 || next_zeroth >= 2048 {
                info!(
                    "Step {}: zeroth={}, eos={:?}, is_eos={}, generated={}",
                    step,
                    next_zeroth,
                    eos_token_id,
                    Some(next_zeroth) == eos_token_id,
                    generated_zeroth.len()
                );
            }
            debug!("Step {}: generated zeroth token {}", step, next_zeroth);

            // Check for EOS
            if Some(next_zeroth) == eos_token_id && generated_zeroth.len() >= min_new_tokens {
                info!(
                    "EOS token generated at step {} (after {} tokens)",
                    step,
                    generated_zeroth.len()
                );
                break;
            }

            // Store hidden state
            all_hidden_states.push(hidden_states.clone());

            generated_zeroth.push(next_zeroth);

            // Predict residual codebooks for this token
            let zeroth_for_predictor = Tensor::new(&[next_zeroth], &self.device)?.unsqueeze(0)?;
            let zeroth_embed_for_predictor = self.embed_codec(&zeroth_for_predictor)?; // [1, 1, hidden_size]
            let residuals = code_predictor.predict_from_hidden(
                &hidden_states,
                &zeroth_for_predictor,
                &zeroth_embed_for_predictor,
                sampling_config.clone(),
            )?;
            let residual_codes: Vec<u32> = residuals.squeeze(0)?.squeeze(0)?.i(1..)?.to_vec1()?;

            // Store frame
            let mut frame = vec![next_zeroth];
            frame.extend(&residual_codes);
            all_frames.push(frame);

            // Update for next iteration
            kv_caches = Some(new_kv_caches);
            position_offset += 1;
            current_zeroth_token = next_zeroth;
            prev_residual_codes = residual_codes;
        }

        info!(
            "Generated {} zeroth tokens, {} frames",
            generated_zeroth.len(),
            all_frames.len()
        );

        // Concatenate hidden states
        let concatenated_hidden = if all_hidden_states.is_empty() {
            Tensor::zeros((1, 0, self.config.hidden_size), DType::F32, &self.device)?
        } else {
            Tensor::cat(&all_hidden_states, 1)?
        };

        Ok((generated_zeroth, all_frames, concatenated_hidden))
    }

    /// Suppress special tokens in logits (set to -inf) except for EOS token.
    ///
    /// In Qwen3-TTS, tokens 0-2047 are valid audio codes.
    /// Tokens 2048+ are special tokens (pad, bos, eos, think markers, etc.)
    /// During generation, we should only allow audio tokens and EOS.
    fn suppress_special_tokens(
        logits: &mut [f32],
        suppress_start: u32,
        suppress_end: u32,
        eos_token_id: Option<u32>,
    ) {
        for i in suppress_start..suppress_end.min(logits.len() as u32) {
            // Don't suppress EOS token
            if Some(i) != eos_token_id {
                logits[i as usize] = f32::NEG_INFINITY;
            }
        }
    }

    /// Forward pass from pre-computed embeddings (for CustomVoice flow).
    ///
    /// # Returns
    /// Tuple of (logits, hidden_states, kv_caches)
    #[allow(clippy::type_complexity)]
    fn forward_embeds(
        &self,
        hidden_states: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[(Tensor, Tensor)]>,
    ) -> Result<(Tensor, Tensor, Vec<(Tensor, Tensor)>)> {
        let mut hidden_states = hidden_states.clone();
        let mut new_kv_caches = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let kv_cache = kv_caches.map(|caches| (&caches[i].0, &caches[i].1));

            let (new_hidden, k_cache, v_cache) =
                layer.forward(&hidden_states, &self.rotary_emb, position_offset, kv_cache)?;

            hidden_states = new_hidden;
            new_kv_caches.push((k_cache, v_cache));
        }

        // Final norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // Project to codec vocabulary
        let logits = self.codec_head.forward(&hidden_states)?;

        Ok((logits, hidden_states, new_kv_caches))
    }

    /// Get text embedding for given token IDs.
    pub fn get_text_embedding(&self, text_ids: &Tensor) -> Result<Tensor> {
        self.embed_text(text_ids)
    }

    /// Get codec embedding for given token IDs.
    pub fn get_codec_embedding(&self, codec_ids: &Tensor) -> Result<Tensor> {
        self.embed_codec(codec_ids)
    }
}

/// A streaming generator for incremental token generation.
pub struct StreamingGenerator<'a> {
    model: &'a Model,
    sampler: Sampler,
    kv_caches: Option<Vec<(Tensor, Tensor)>>,
    position_offset: usize,
    eos_token_id: Option<u32>,
    max_tokens: usize,
    generated_count: usize,
    finished: bool,
}

impl<'a> StreamingGenerator<'a> {
    /// Create a new streaming generator.
    pub fn new(
        model: &'a Model,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
        max_tokens: usize,
    ) -> Self {
        Self {
            model,
            sampler: Sampler::new(sampling_config),
            kv_caches: None,
            position_offset: 0,
            eos_token_id,
            max_tokens,
            generated_count: 0,
            finished: false,
        }
    }

    /// Initialize with input tokens (prefill).
    pub fn prefill(&mut self, input_ids: &[u32]) -> Result<()> {
        let input_tensor = Tensor::new(input_ids, self.model.device())?.unsqueeze(0)?;

        let (_, new_kv_caches) = self.model.forward(&input_tensor, 0, None)?;

        self.kv_caches = Some(new_kv_caches);
        self.position_offset = input_ids.len();

        Ok(())
    }

    /// Generate the next token.
    pub fn next(&mut self, current_token: u32) -> Result<Option<u32>> {
        if self.finished || self.generated_count >= self.max_tokens {
            return Ok(None);
        }

        let input_tensor = Tensor::new(&[current_token], self.model.device())?.unsqueeze(0)?;

        let (logits, new_kv_caches) = self.model.forward(
            &input_tensor,
            self.position_offset,
            self.kv_caches.as_deref(),
        )?;

        // Get logits for the last position
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
        let logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

        // Sample next token
        let next_token = self.sampler.sample(&logits_vec);

        // Check for EOS
        if Some(next_token) == self.eos_token_id {
            self.finished = true;
            return Ok(None);
        }

        // Update state
        self.kv_caches = Some(new_kv_caches);
        self.position_offset += 1;
        self.generated_count += 1;

        Ok(Some(next_token))
    }

    /// Check if generation is finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get the number of tokens generated so far.
    pub fn generated_count(&self) -> usize {
        self.generated_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AcousticModelConfig {
        AcousticModelConfig::tiny()
    }

    // Note: Full model tests require weight files
    // These tests verify the structure and API

    #[test]
    fn test_config_creation() {
        let config = test_config();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_streaming_generator_state() {
        // Test that generator state is properly initialized
        let config = test_config();
        let sampling_config = SamplingConfig::default();

        // We can't test full generation without weights,
        // but we can verify the structure
        assert_eq!(sampling_config.temperature, 1.0);
        assert_eq!(config.codec_vocab_size, 100);
    }
}
