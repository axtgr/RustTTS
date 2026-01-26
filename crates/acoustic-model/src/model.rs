//! Main acoustic model implementation.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};
use tracing::{debug, info, instrument, warn};

use crate::config::AcousticModelConfig;
use crate::layers::{RmsNorm, RotaryEmbedding, TextProjection, TransformerBlock};
use crate::sampling::{Sampler, SamplingConfig};

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

    /// Forward pass with explicit text and codec token handling.
    ///
    /// This is the proper forward pass for Qwen3-TTS that handles
    /// mixed text/codec sequences correctly.
    #[instrument(skip(self, text_ids, codec_ids, kv_caches))]
    pub fn forward_mixed(
        &self,
        text_ids: Option<&Tensor>,
        codec_ids: Option<&Tensor>,
        position_offset: usize,
        kv_caches: Option<&[(Tensor, Tensor)]>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (logits, _, new_kv_caches) =
            self.forward_mixed_with_hidden(text_ids, codec_ids, position_offset, kv_caches)?;
        Ok((logits, new_kv_caches))
    }

    /// Forward pass that also returns hidden states (before codec_head).
    ///
    /// This is needed for CodePredictor which uses the hidden states
    /// to predict residual codebooks (1-15).
    ///
    /// # Returns
    /// Tuple of (logits, hidden_states, kv_caches)
    /// - logits: [batch, seq_len, codec_vocab_size]
    /// - hidden_states: [batch, seq_len, hidden_size] - after final norm
    /// - kv_caches: Updated KV caches for each layer
    #[instrument(skip(self, text_ids, codec_ids, kv_caches))]
    pub fn forward_mixed_with_hidden(
        &self,
        text_ids: Option<&Tensor>,
        codec_ids: Option<&Tensor>,
        position_offset: usize,
        kv_caches: Option<&[(Tensor, Tensor)]>,
    ) -> Result<ForwardWithHiddenOutput> {
        // Embed tokens based on type
        let hidden_states = match (text_ids, codec_ids) {
            (Some(text), None) => self.embed_text(text)?,
            (None, Some(codec)) => self.embed_codec(codec)?,
            (Some(text), Some(codec)) => {
                let text_hidden = self.embed_text(text)?;
                let codec_hidden = self.embed_codec(codec)?;
                Tensor::cat(&[text_hidden, codec_hidden], 1)?
            }
            (None, None) => {
                return Err(candle_core::Error::Msg(
                    "Must provide either text_ids or codec_ids".to_string(),
                ));
            }
        };

        let mut hidden_states = hidden_states;

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

        Ok((logits, hidden_states, new_kv_caches))
    }

    /// Generate acoustic tokens autoregressively from text tokens.
    ///
    /// Uses forward_mixed to properly handle text prefill and codec generation.
    /// Returns only the zeroth codebook tokens.
    ///
    /// # Arguments
    /// - `text_ids`: Text tokens (including tts_bos, but NOT codec_bos)
    /// - `codec_bos_id`: The codec BOS token ID to start generation
    /// - `max_new_tokens`: Maximum number of tokens to generate
    /// - `sampling_config`: Sampling configuration
    /// - `eos_token_id`: EOS token ID to stop generation
    #[instrument(skip(self, text_ids, sampling_config))]
    pub fn generate(
        &self,
        text_ids: &[u32],
        codec_bos_id: u32,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        let (tokens, _) = self.generate_with_hidden(
            text_ids,
            codec_bos_id,
            max_new_tokens,
            sampling_config,
            eos_token_id,
        )?;
        Ok(tokens)
    }

    /// Generate acoustic tokens and return hidden states for CodePredictor.
    ///
    /// This method generates zeroth codebook tokens and also returns the
    /// hidden states from each generation step, which can be used by
    /// CodePredictor to predict residual codebooks (1-15).
    ///
    /// # Arguments
    /// - `text_ids`: Text tokens (including tts_bos, but NOT codec_bos)
    /// - `codec_bos_id`: The codec BOS token ID to start generation
    /// - `max_new_tokens`: Maximum number of tokens to generate
    /// - `sampling_config`: Sampling configuration
    /// - `eos_token_id`: EOS token ID to stop generation
    ///
    /// # Returns
    /// Tuple of (tokens, hidden_states)
    /// - tokens: Vec<u32> - generated zeroth codebook tokens (NOT including codec_bos)
    /// - hidden_states: Tensor [1, num_tokens, hidden_size] - concatenated hidden states
    #[instrument(
        skip(self, text_ids, sampling_config),
        fields(max_new_tokens, eos_token_id)
    )]
    pub fn generate_with_hidden(
        &self,
        text_ids: &[u32],
        codec_bos_id: u32,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, Tensor)> {
        info!(
            "Generating up to {} tokens from {} text tokens + codec_bos={} (with hidden states)",
            max_new_tokens,
            text_ids.len(),
            codec_bos_id
        );

        let mut sampler = Sampler::new(sampling_config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut all_hidden_states: Vec<Tensor> = Vec::with_capacity(max_new_tokens);

        // Step 1: Prefill with text tokens only (using text_embedding)
        let text_tensor = Tensor::new(text_ids, &self.device)?.unsqueeze(0)?; // [1, text_len]
        let (_, _, text_kv_caches) =
            self.forward_mixed_with_hidden(Some(&text_tensor), None, 0, None)?;

        let mut position_offset = text_ids.len();

        // Step 2: Forward codec_bos_id (using codec_embedding) to get first logits
        let codec_bos_tensor = Tensor::new(&[codec_bos_id], &self.device)?.unsqueeze(0)?;
        let (logits, hidden_states, new_kv_caches) = self.forward_mixed_with_hidden(
            None,
            Some(&codec_bos_tensor),
            position_offset,
            Some(&text_kv_caches),
        )?;

        let mut kv_caches: Option<Vec<(Tensor, Tensor)>> = Some(new_kv_caches);
        position_offset += 1;

        // Sample first codec token from logits after codec_bos
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
        let logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;
        let mut current_token = sampler.sample(&logits_vec);

        debug!("Step 0: sampled token {} after codec_bos", current_token);

        // Check for EOS
        if Some(current_token) == eos_token_id {
            info!("EOS token generated at first step");
            let empty_hidden =
                Tensor::zeros((1, 0, self.config.hidden_size), DType::F32, &self.device)?;
            return Ok((generated, empty_hidden));
        }
        generated.push(current_token);

        // Store hidden state for codec_bos position (needed for CodePredictor alignment)
        all_hidden_states.push(hidden_states.clone());

        // Generate remaining tokens
        for step in 1..max_new_tokens {
            // Create codec token tensor
            let codec_tensor = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;

            // Forward pass with codec token
            let (logits, hidden_states, new_kv_caches) = self.forward_mixed_with_hidden(
                None,
                Some(&codec_tensor),
                position_offset,
                kv_caches.as_deref(),
            )?;

            // Store hidden state for this step
            all_hidden_states.push(hidden_states.clone());

            // Get logits for the last position
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

            // Sample next token
            let next_token = sampler.sample(&logits_vec);

            // Log every 50 steps or when close to EOS
            if step % 50 == 0 || next_token >= 2145 {
                info!(
                    "Step {}: token={}, eos_token={:?}, is_eos={}",
                    step,
                    next_token,
                    eos_token_id,
                    Some(next_token) == eos_token_id
                );
            }
            debug!("Step {}: generated token {}", step, next_token);

            // Check for EOS
            if Some(next_token) == eos_token_id {
                info!("EOS token generated at step {}", step);
                break;
            }

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

        debug!("Hidden states shape: {:?}", concatenated_hidden.dims());

        Ok((generated, concatenated_hidden))
    }

    /// Generate acoustic tokens from combined embeddings (Qwen3-TTS CustomVoice format).
    ///
    /// In Qwen3-TTS, the input to the talker is a **sum** of text_embedding and codec_embedding,
    /// not a concatenation. This method generates from pre-computed combined embeddings.
    ///
    /// # Arguments
    /// - `combined_embeds`: Pre-computed embeddings [1, seq_len, hidden_size] = text_embed + codec_embed
    /// - `max_new_tokens`: Maximum number of tokens to generate
    /// - `sampling_config`: Sampling configuration
    /// - `eos_token_id`: EOS token ID to stop generation
    ///
    /// # Returns
    /// Tuple of (tokens, hidden_states)
    #[instrument(skip(self, combined_embeds, sampling_config), fields(max_new_tokens))]
    pub fn generate_from_embeds(
        &self,
        combined_embeds: &Tensor,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, Tensor)> {
        let seq_len = combined_embeds.dim(1)?;
        info!(
            "Generating up to {} tokens from {} prefill embeddings",
            max_new_tokens, seq_len
        );

        let mut sampler = Sampler::new(sampling_config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut all_hidden_states: Vec<Tensor> = Vec::with_capacity(max_new_tokens);

        // Step 1: Prefill with combined embeddings
        let (logits, hidden_states, kv_caches) = self.forward_embeds(combined_embeds, 0, None)?;

        let mut position_offset = seq_len;
        let mut kv_caches: Option<Vec<(Tensor, Tensor)>> = Some(kv_caches);

        // Sample first codec token from last position's logits
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
        let logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;
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
        all_hidden_states.push(hidden_states.i((.., seq_len - 1.., ..))?.clone());

        // Generate remaining tokens
        for step in 1..max_new_tokens {
            // Embed current codec token
            let codec_tensor = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;
            let codec_embed = self.embed_codec(&codec_tensor)?;

            // Forward pass with codec embedding
            let (logits, hidden_states, new_kv_caches) =
                self.forward_embeds(&codec_embed, position_offset, kv_caches.as_deref())?;

            // Store hidden state for this step
            all_hidden_states.push(hidden_states.clone());

            // Get logits for the last position
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

            // Sample next token
            let next_token = sampler.sample(&logits_vec);

            // Log every 50 steps or when close to EOS
            if step % 50 == 0 || next_token >= 2145 {
                info!(
                    "Step {}: token={}, eos_token={:?}, is_eos={}",
                    step,
                    next_token,
                    eos_token_id,
                    Some(next_token) == eos_token_id
                );
            }
            debug!("Step {}: generated token {}", step, next_token);

            // Check for EOS
            if Some(next_token) == eos_token_id {
                info!("EOS token generated at step {}", step);
                break;
            }

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

    /// Forward pass from pre-computed embeddings (for CustomVoice flow).
    ///
    /// # Returns
    /// Tuple of (logits, hidden_states, kv_caches)
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
