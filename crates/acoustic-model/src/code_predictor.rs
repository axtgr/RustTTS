//! Code Predictor (MTP - Multi-Token Prediction) module.
//!
//! This module predicts the residual codebooks (1-15) from the zeroth codebook
//! predicted by the main Talker model. This enables parallel prediction of all
//! 16 codebooks for ultra-low-latency streaming.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding, linear};
use tracing::{debug, info, instrument};

use crate::config::CodePredictorConfig;
use crate::layers::{RmsNorm, RotaryEmbedding, TransformerBlock};
use crate::sampling::{Sampler, SamplingConfig};

/// Code Predictor model for multi-codebook prediction.
///
/// Given the zeroth (semantic) codebook tokens, predicts all residual
/// acoustic codebook tokens (1 through num_code_groups-1).
#[derive(Debug)]
pub struct CodePredictor {
    /// Embedding for codebook tokens (shared across all codebooks).
    embed_tokens: Embedding,
    /// Transformer layers.
    layers: Vec<TransformerBlock>,
    /// Final layer normalization.
    norm: RmsNorm,
    /// Output heads for each codebook (predicts next token for each group).
    lm_heads: Vec<candle_nn::Linear>,
    /// Rotary position embeddings.
    rotary_emb: RotaryEmbedding,
    /// Model configuration.
    config: CodePredictorConfig,
    /// Device.
    device: Device,
}

impl CodePredictor {
    /// Load CodePredictor from safetensors weights.
    #[instrument(skip(path, config, device), fields(num_layers = config.num_layers))]
    pub fn load(
        path: impl AsRef<std::path::Path>,
        config: CodePredictorConfig,
        device: &Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading CodePredictor from {}", path.display());

        let tensors = candle_core::safetensors::load(path, device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

        Self::from_vb(vb, config, device)
    }

    /// Create CodePredictor from VarBuilder.
    pub fn from_vb(vb: VarBuilder, config: CodePredictorConfig, device: &Device) -> Result<Self> {
        let vb_model = vb.pp("code_predictor");

        // Shared embedding for all codebooks
        // Input vocab is codebook_size (2048) per group
        let embed_tokens = embedding(
            config.codebook_size,
            config.hidden_size,
            vb_model.pp("embed_tokens"),
        )?;

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);

        // Create a temporary AcousticModelConfig for transformer blocks
        let acoustic_config = crate::config::AcousticModelConfig {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            num_layers: config.num_layers,
            intermediate_size: config.intermediate_size,
            head_dim: config.head_dim,
            text_vocab_size: 0, // Not used
            codec_vocab_size: config.codebook_size,
            num_code_groups: config.num_code_groups,
            codebook_size: config.codebook_size,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
            tts_bos_token_id: 0,
            tts_eos_token_id: 0,
            tts_pad_token_id: 0,
            codec_bos_id: 0,
            codec_eos_id: 0,
            codec_pad_id: 0,
        };

        for i in 0..config.num_layers {
            debug!(
                "Loading CodePredictor layer {}/{}",
                i + 1,
                config.num_layers
            );
            let layer =
                TransformerBlock::new(&acoustic_config, vb_model.pp(format!("layers.{i}")))?;
            layers.push(layer);
        }

        // Final norm
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb_model.pp("norm"))?;

        // Separate LM head for each residual codebook (groups 1 to num_code_groups-1)
        let num_residual_groups = config.num_code_groups - 1;
        let mut lm_heads = Vec::with_capacity(num_residual_groups);
        for i in 0..num_residual_groups {
            let head = linear(
                config.hidden_size,
                config.codebook_size,
                vb_model.pp(format!("lm_head.{i}")),
            )?;
            lm_heads.push(head);
        }

        // Rotary embeddings
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            device,
        )?;

        info!(
            "CodePredictor loaded: {} layers, {} groups",
            config.num_layers, config.num_code_groups
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_heads,
            rotary_emb,
            config,
            device: device.clone(),
        })
    }

    /// Create CodePredictor with random weights (for testing).
    pub fn new_random(config: CodePredictorConfig, device: &Device) -> Result<Self> {
        // We need to create tensors manually for testing
        let embed_tokens_weight = Tensor::randn(
            0.0f32,
            0.02,
            (config.codebook_size, config.hidden_size),
            device,
        )?;
        let embed_tokens = Embedding::new(embed_tokens_weight, config.hidden_size);

        // For testing, we'll create minimal layers
        let layers = Vec::new(); // Empty for mock

        let norm = RmsNorm::new_ones(config.hidden_size, config.rms_norm_eps, device)?;

        // Create random LM heads
        let num_residual_groups = config.num_code_groups - 1;
        let mut lm_heads = Vec::with_capacity(num_residual_groups);
        for _ in 0..num_residual_groups {
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (config.codebook_size, config.hidden_size),
                device,
            )?;
            let bias = Tensor::zeros((config.codebook_size,), DType::F32, device)?;
            let head = candle_nn::Linear::new(weight, Some(bias));
            lm_heads.push(head);
        }

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_heads,
            rotary_emb,
            config,
            device: device.clone(),
        })
    }

    /// Get configuration.
    pub fn config(&self) -> &CodePredictorConfig {
        &self.config
    }

    /// Get device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Predict all residual codebooks from zeroth codebook.
    ///
    /// # Arguments
    /// * `zeroth_codes` - Tokens from the zeroth (semantic) codebook [batch, seq_len]
    /// * `sampling_config` - Sampling configuration
    ///
    /// # Returns
    /// All codebook tokens [batch, seq_len, num_code_groups]
    /// where index 0 is the input zeroth_codes
    #[instrument(skip(self, zeroth_codes, sampling_config))]
    pub fn predict(
        &self,
        zeroth_codes: &Tensor,
        sampling_config: SamplingConfig,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = zeroth_codes.dims2()?;

        // Initialize output tensor with all codebooks
        // Shape: [batch, seq_len, num_groups]
        let mut all_codes = vec![zeroth_codes.clone()];

        // Embed zeroth codes
        let hidden_states = self.embed_tokens.forward(zeroth_codes)?;

        // Pass through transformer layers (if any)
        let mut hidden = hidden_states;
        for (i, layer) in self.layers.iter().enumerate() {
            let (new_hidden, _, _) = layer.forward(&hidden, &self.rotary_emb, 0, None)?;
            hidden = new_hidden;
            debug!("CodePredictor layer {} done", i);
        }

        // Apply final norm
        let hidden = self.norm.forward(&hidden)?;

        // Predict each residual codebook in parallel
        let mut sampler = Sampler::new(sampling_config);

        for (group_idx, lm_head) in self.lm_heads.iter().enumerate() {
            // Get logits for this codebook
            let logits = lm_head.forward(&hidden)?; // [batch, seq_len, codebook_size]

            // Sample tokens for each position
            let mut group_codes = Vec::with_capacity(batch_size * seq_len);

            for b in 0..batch_size {
                for s in 0..seq_len {
                    let pos_logits = logits.i((b, s, ..))?;
                    let logits_vec: Vec<f32> = pos_logits.to_vec1()?;
                    let token = sampler.sample(&logits_vec);
                    group_codes.push(token);
                }
            }

            let group_tensor = Tensor::new(group_codes.as_slice(), &self.device)?
                .reshape((batch_size, seq_len))?;
            all_codes.push(group_tensor);

            debug!("Predicted codebook group {}", group_idx + 1);
        }

        // Stack all codebooks: [batch, seq_len, num_groups]
        let stacked: Vec<Tensor> = all_codes
            .iter()
            .map(|t| t.unsqueeze(2))
            .collect::<Result<Vec<_>>>()?;

        Tensor::cat(&stacked, 2)
    }

    /// Predict residual codebooks for a single frame (streaming mode).
    ///
    /// # Arguments
    /// * `zeroth_code` - Single token from zeroth codebook
    /// * `sampling_config` - Sampling configuration
    ///
    /// # Returns
    /// All codebook tokens for this frame [num_groups]
    pub fn predict_frame(
        &self,
        zeroth_code: u32,
        sampling_config: SamplingConfig,
    ) -> Result<Vec<u32>> {
        let input = Tensor::new(&[zeroth_code], &self.device)?.unsqueeze(0)?;
        let all_codes = self.predict(&input, sampling_config)?;

        // Extract the single frame
        let codes: Vec<u32> = all_codes.squeeze(0)?.squeeze(0)?.to_vec1()?;
        Ok(codes)
    }
}

/// Multi-codebook output container.
#[derive(Debug, Clone)]
pub struct MultiCodebookOutput {
    /// Tokens for each codebook group.
    /// codes[i] contains tokens for codebook i.
    pub codes: Vec<Vec<u32>>,
    /// Number of frames (tokens per codebook).
    pub num_frames: usize,
    /// Number of codebook groups.
    pub num_groups: usize,
}

impl MultiCodebookOutput {
    /// Create new multi-codebook output.
    pub fn new(num_groups: usize) -> Self {
        Self {
            codes: vec![Vec::new(); num_groups],
            num_frames: 0,
            num_groups,
        }
    }

    /// Add a frame of codes.
    pub fn add_frame(&mut self, frame_codes: &[u32]) {
        assert_eq!(frame_codes.len(), self.num_groups);
        for (i, &code) in frame_codes.iter().enumerate() {
            self.codes[i].push(code);
        }
        self.num_frames += 1;
    }

    /// Get codes for a specific codebook group.
    pub fn get_group(&self, group_idx: usize) -> &[u32] {
        &self.codes[group_idx]
    }

    /// Get all codes as a flat interleaved vector.
    /// Format: [frame0_group0, frame0_group1, ..., frame1_group0, ...]
    pub fn to_interleaved(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.num_frames * self.num_groups);
        for frame_idx in 0..self.num_frames {
            for group_idx in 0..self.num_groups {
                result.push(self.codes[group_idx][frame_idx]);
            }
        }
        result
    }

    /// Get all codes as grouped vectors.
    /// Format: [[group0_codes], [group1_codes], ...]
    pub fn to_grouped(&self) -> Vec<Vec<u32>> {
        self.codes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CodePredictorConfig {
        CodePredictorConfig::tiny()
    }

    #[test]
    fn test_multi_codebook_output() {
        let mut output = MultiCodebookOutput::new(4);

        output.add_frame(&[1, 2, 3, 4]);
        output.add_frame(&[5, 6, 7, 8]);

        assert_eq!(output.num_frames, 2);
        assert_eq!(output.get_group(0), &[1, 5]);
        assert_eq!(output.get_group(1), &[2, 6]);

        let interleaved = output.to_interleaved();
        assert_eq!(interleaved, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_code_predictor_config() {
        let config = test_config();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.num_code_groups, 4);
    }

    #[test]
    fn test_code_predictor_creation() {
        let device = Device::Cpu;
        let config = test_config();

        let predictor = CodePredictor::new_random(config.clone(), &device).unwrap();

        assert_eq!(predictor.config().num_code_groups, 4);
        assert_eq!(predictor.lm_heads.len(), 3); // num_groups - 1
    }

    #[test]
    fn test_code_predictor_predict_frame() {
        let device = Device::Cpu;
        let config = test_config();

        let predictor = CodePredictor::new_random(config.clone(), &device).unwrap();
        let sampling = SamplingConfig::greedy();

        let frame = predictor.predict_frame(10, sampling).unwrap();

        assert_eq!(frame.len(), config.num_code_groups);
    }
}
