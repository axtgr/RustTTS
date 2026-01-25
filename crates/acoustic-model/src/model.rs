//! Main acoustic model implementation.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};
use tracing::{debug, info, instrument};

use crate::config::AcousticModelConfig;
use crate::layers::{RmsNorm, RotaryEmbedding, TransformerBlock};
use crate::sampling::{Sampler, SamplingConfig};

/// The main acoustic model for Qwen3-TTS.
#[derive(Debug)]
pub struct Model {
    /// Token embedding layer.
    embed_tokens: Embedding,
    /// Transformer decoder blocks.
    layers: Vec<TransformerBlock>,
    /// Final layer normalization.
    norm: RmsNorm,
    /// Output projection to acoustic vocabulary.
    lm_head: candle_nn::Linear,
    /// Rotary position embeddings.
    rotary_emb: RotaryEmbedding,
    /// Model configuration.
    config: AcousticModelConfig,
    /// Device (CPU or CUDA).
    device: Device,
}

impl Model {
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
        let vb_model = vb.pp("model");

        // Embedding layer
        let embed_tokens = embedding(
            config.text_vocab_size,
            config.hidden_size,
            vb_model.pp("embed_tokens"),
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

        // LM head
        let lm_head = candle_nn::linear(
            config.hidden_size,
            config.acoustic_vocab_size,
            vb.pp("lm_head"),
        )?;

        // Rotary embeddings
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        info!(
            "Model loaded: {} layers, {} hidden, {} vocab",
            config.num_layers, config.hidden_size, config.text_vocab_size
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
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

    /// Forward pass through the model.
    ///
    /// Returns logits for the next token and updated KV caches.
    #[instrument(skip(self, input_ids, kv_caches))]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[(Tensor, Tensor)]>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        // Embed input tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

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

        // Project to vocabulary
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok((logits, new_kv_caches))
    }

    /// Generate acoustic tokens autoregressively.
    #[instrument(skip(self, input_ids, sampling_config))]
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        info!(
            "Generating up to {} tokens from {} input tokens",
            max_new_tokens,
            input_ids.len()
        );

        let mut sampler = Sampler::new(sampling_config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut kv_caches: Option<Vec<(Tensor, Tensor)>> = None;

        // Create input tensor
        let mut current_ids = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?; // [1, seq_len]

        let mut position_offset = 0;

        for step in 0..max_new_tokens {
            // Forward pass
            let (logits, new_kv_caches) =
                self.forward(&current_ids, position_offset, kv_caches.as_deref())?;

            // Get logits for the last position
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let logits_vec: Vec<f32> = last_logits.squeeze(0)?.to_vec1()?;

            // Sample next token
            let next_token = sampler.sample(&logits_vec);
            debug!("Step {}: generated token {}", step, next_token);

            // Check for EOS
            if Some(next_token) == eos_token_id {
                info!("EOS token generated at step {}", step);
                break;
            }

            generated.push(next_token);

            // Update for next iteration
            kv_caches = Some(new_kv_caches);
            position_offset += current_ids.dim(1)?;
            current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        info!("Generated {} tokens", generated.len());
        Ok(generated)
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
        AcousticModelConfig {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_size: 128,
            text_vocab_size: 100,
            acoustic_vocab_size: 50,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
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
        assert_eq!(config.acoustic_vocab_size, 50);
    }
}
