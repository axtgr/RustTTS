//! Code Predictor (MTP - Multi-Token Prediction) module.
//!
//! This module predicts the residual codebooks (1-15) from the zeroth codebook
//! predicted by the main Talker model. This enables parallel prediction of all
//! 16 codebooks for ultra-low-latency streaming.
//!
//! Qwen3-TTS CodePredictor architecture:
//! - 15 separate codec_embeddings for codebooks 1-15
//! - 5 transformer layers with QK-Norm
//! - 15 separate lm_heads for predicting codebooks 1-15
//! - Takes hidden states from Talker as input

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};
use tracing::{debug, info, instrument, warn};

use crate::config::CodePredictorConfig;
use crate::layers::{RmsNorm, RotaryEmbedding, TransformerBlock};
use crate::sampling::{Sampler, SamplingConfig};

/// Code Predictor model for multi-codebook prediction.
///
/// Given the zeroth (semantic) codebook tokens from Talker, predicts all residual
/// acoustic codebook tokens (1 through num_code_groups-1) in parallel.
///
/// Qwen3-TTS structure:
/// - `codec_embedding.{0-14}`: Separate embeddings for each residual codebook
/// - `layers.{0-4}`: 5 transformer layers
/// - `lm_head.{0-14}`: Separate output heads for each residual codebook
#[derive(Debug)]
pub struct CodePredictor {
    /// Separate embeddings for each residual codebook [codebook_size, hidden_size].
    /// codec_embeddings[i] is for codebook i+1 (residual codebook).
    codec_embeddings: Vec<Embedding>,
    /// Transformer layers.
    layers: Vec<TransformerBlock>,
    /// Final layer normalization.
    norm: RmsNorm,
    /// Output heads for each residual codebook (predicts codebook i+1).
    /// lm_heads[i] outputs logits for codebook i+1.
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
    ///
    /// Qwen3-TTS path: `talker.code_predictor.model.*` and `talker.code_predictor.lm_head.*`
    pub fn from_vb(vb: VarBuilder, config: CodePredictorConfig, device: &Device) -> Result<Self> {
        // Qwen3-TTS: talker.code_predictor.model.* and talker.code_predictor.lm_head.*
        let vb_predictor = vb.pp("talker").pp("code_predictor");
        let vb_model = vb_predictor.pp("model");

        let num_residual_groups = config.num_code_groups - 1; // 15 for Qwen3-TTS

        // Separate codec embeddings for each residual codebook
        // Qwen3-TTS: talker.code_predictor.model.codec_embedding.{0-14}.weight [2048, 1024]
        let mut codec_embeddings = Vec::with_capacity(num_residual_groups);
        for i in 0..num_residual_groups {
            let emb = embedding(
                config.codebook_size,
                config.hidden_size,
                vb_model.pp(format!("codec_embedding.{i}")),
            )?;
            codec_embeddings.push(emb);
            debug!("Loaded codec_embedding.{}", i);
        }

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);

        // Create a temporary AcousticModelConfig for transformer blocks
        let acoustic_config = crate::config::AcousticModelConfig {
            hidden_size: config.hidden_size,
            embedding_dim: config.hidden_size, // Same as hidden_size for code predictor
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
        // Qwen3-TTS: talker.code_predictor.lm_head.{0-14}.weight [2048, 1024]
        let mut lm_heads = Vec::with_capacity(num_residual_groups);
        for i in 0..num_residual_groups {
            let head = candle_nn::linear_no_bias(
                config.hidden_size,
                config.codebook_size,
                vb_predictor.pp(format!("lm_head.{i}")),
            )?;
            lm_heads.push(head);
            debug!("Loaded lm_head.{}", i);
        }

        // Rotary embeddings
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            device,
        )?;

        info!(
            "CodePredictor loaded: {} layers, {} residual codebooks, hidden={}",
            config.num_layers, num_residual_groups, config.hidden_size
        );

        Ok(Self {
            codec_embeddings,
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
        let num_residual_groups = config.num_code_groups - 1;

        // Create random embeddings for each residual codebook
        let mut codec_embeddings = Vec::with_capacity(num_residual_groups);
        for _ in 0..num_residual_groups {
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (config.codebook_size, config.hidden_size),
                device,
            )?;
            codec_embeddings.push(Embedding::new(weight, config.hidden_size));
        }

        // For testing, we'll create minimal layers
        let layers = Vec::new(); // Empty for mock

        let norm = RmsNorm::new_ones(config.hidden_size, config.rms_norm_eps, device)?;

        // Create random LM heads (no bias for Qwen3-TTS)
        let mut lm_heads = Vec::with_capacity(num_residual_groups);
        for _ in 0..num_residual_groups {
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (config.codebook_size, config.hidden_size),
                device,
            )?;
            let head = candle_nn::Linear::new(weight, None);
            lm_heads.push(head);
        }

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            device,
        )?;

        Ok(Self {
            codec_embeddings,
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

    /// Get embedding for a specific residual codebook.
    ///
    /// This is used during generation to get embeddings for predicted tokens
    /// that will be summed together for the next step.
    ///
    /// # Arguments
    /// * `codebook_idx` - Index 0-14 for residual codebooks 1-15
    /// * `token_ids` - Token IDs to embed [batch, seq_len]
    ///
    /// # Returns
    /// Embeddings [batch, seq_len, hidden_size]
    pub fn get_codec_embedding(&self, codebook_idx: usize, token_ids: &Tensor) -> Result<Tensor> {
        if codebook_idx >= self.codec_embeddings.len() {
            return Err(candle_core::Error::Msg(format!(
                "codebook_idx {} out of range (max {})",
                codebook_idx,
                self.codec_embeddings.len() - 1
            )));
        }
        self.codec_embeddings[codebook_idx].forward(token_ids)
    }

    /// Get embeddings for all residual codebooks at once.
    ///
    /// Given tokens for all 15 residual codebooks, returns the sum of their embeddings.
    /// This is used during generation: `inputs_embeds = sum(codec_embeddings[i](tokens[i]))`.
    ///
    /// # Arguments
    /// * `residual_codes` - Tokens for codebooks 1-15, shape [batch, 15] or [batch, seq, 15]
    ///
    /// # Returns
    /// Sum of all embeddings [batch, 1, hidden_size] or [batch, seq, hidden_size]
    pub fn sum_residual_embeddings(&self, residual_codes: &Tensor) -> Result<Tensor> {
        let dims = residual_codes.dims();
        let num_residual = self.codec_embeddings.len(); // 15

        match dims.len() {
            // [batch, 15] - single position
            2 => {
                let (_batch, num_codes) = residual_codes.dims2()?;
                if num_codes != num_residual {
                    return Err(candle_core::Error::Msg(format!(
                        "Expected {} residual codes, got {}",
                        num_residual, num_codes
                    )));
                }

                let mut sum: Option<Tensor> = None;
                for i in 0..num_residual {
                    let codes_i = residual_codes.i((.., i))?.unsqueeze(1)?; // [batch, 1]
                    let emb = self.codec_embeddings[i].forward(&codes_i)?; // [batch, 1, hidden]
                    sum = Some(match sum {
                        None => emb,
                        Some(s) => (&s + &emb)?,
                    });
                }
                sum.ok_or_else(|| candle_core::Error::Msg("No embeddings to sum".into()))
            }
            // [batch, seq, 15] - multiple positions
            3 => {
                let (_batch, _seq, num_codes) = residual_codes.dims3()?;
                if num_codes != num_residual {
                    return Err(candle_core::Error::Msg(format!(
                        "Expected {} residual codes, got {}",
                        num_residual, num_codes
                    )));
                }

                let mut sum: Option<Tensor> = None;
                for i in 0..num_residual {
                    let codes_i = residual_codes.i((.., .., i))?; // [batch, seq]
                    let emb = self.codec_embeddings[i].forward(&codes_i)?; // [batch, seq, hidden]
                    sum = Some(match sum {
                        None => emb,
                        Some(s) => (&s + &emb)?,
                    });
                }
                sum.ok_or_else(|| candle_core::Error::Msg("No embeddings to sum".into()))
            }
            _ => Err(candle_core::Error::Msg(format!(
                "Expected 2D or 3D tensor, got {}D",
                dims.len()
            ))),
        }
    }

    /// Predict all residual codebooks from hidden states and zeroth codebook.
    ///
    /// This implements the Python SDK's CodePredictor autoregressive generation:
    /// 1. Prefill with [past_hidden, zeroth_embed] (2 tokens)
    /// 2. Generate 14 more tokens autoregressively, each using the embedding
    ///    of the previously predicted token
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states from Talker's LAST layer [batch, 1, hidden_size]
    /// * `zeroth_codes` - Token from the zeroth (semantic) codebook [batch, 1]
    /// * `zeroth_embed` - Embedding of zeroth codebook token [batch, 1, hidden_size]
    /// * `sampling_config` - Sampling configuration
    ///
    /// # Returns
    /// All codebook tokens [batch, 1, num_code_groups]
    /// where index 0 is the input zeroth_codes
    #[instrument(skip(self, hidden_states, zeroth_codes, zeroth_embed, sampling_config))]
    pub fn predict_from_hidden(
        &self,
        hidden_states: &Tensor,
        zeroth_codes: &Tensor,
        zeroth_embed: &Tensor,
        sampling_config: SamplingConfig,
    ) -> Result<Tensor> {
        let (batch_size, _, _) = hidden_states.dims3()?;

        // Initialize output with zeroth codebook
        let mut all_codes = vec![zeroth_codes.clone()];

        // ===== PREFILL STAGE =====
        // Python SDK: inputs_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)
        // past_hidden = hidden_states from Talker's last layer
        // last_id_hidden = embedding of zeroth codebook token
        let prefill_embeds = Tensor::cat(&[hidden_states, zeroth_embed], 1)?; // [batch, 2, hidden]
        debug!(
            "CodePredictor prefill: embeds shape {:?}",
            prefill_embeds.dims()
        );

        // Pass through transformer layers with KV cache
        let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut current_hidden = prefill_embeds;

        for (i, layer) in self.layers.iter().enumerate() {
            let (new_hidden, k_cache, v_cache) =
                layer.forward(&current_hidden, &self.rotary_emb, 0, None)?;
            kv_caches[i] = Some((k_cache, v_cache));
            current_hidden = new_hidden;
        }

        // Apply final norm
        let normed_hidden = self.norm.forward(&current_hidden)?;

        // Get the last token's hidden state (after zeroth embedding)
        let last_hidden = normed_hidden.i((.., 1.., ..))?; // [batch, 1, hidden]

        // Predict first residual codebook (codebook 1)
        // Python: logits = self.lm_head[generation_steps](hidden_states)
        // For prefill, generation_steps = inputs_embeds.shape[1] - 2 = 0
        let logits = self.lm_heads[0].forward(&last_hidden)?; // [batch, 1, codebook_size]

        let mut sampler = Sampler::new(sampling_config);
        let mut predicted_code = self.sample_from_logits(&logits, &mut sampler, batch_size)?;
        all_codes.push(predicted_code.clone());
        debug!(
            "CodePredictor: codebook 1 = {:?}",
            predicted_code.to_vec2::<u32>()?
        );

        // ===== GENERATION STAGE =====
        // Generate codebooks 2-15 autoregressively
        let mut position_offset = 2; // We've processed 2 tokens in prefill

        for gen_step in 1..15 {
            // Python: inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
            // generation_steps = gen_step, so we use codec_embedding[gen_step - 1]
            let input_embed = self.codec_embeddings[gen_step - 1].forward(&predicted_code)?; // [batch, 1, hidden]

            // Pass through transformer layers with KV cache
            let mut current_hidden = input_embed;

            for (i, layer) in self.layers.iter().enumerate() {
                let kv_ref = kv_caches[i].as_ref().map(|(k, v)| (k, v));
                let (new_hidden, k_cache, v_cache) =
                    layer.forward(&current_hidden, &self.rotary_emb, position_offset, kv_ref)?;
                kv_caches[i] = Some((k_cache, v_cache));
                current_hidden = new_hidden;
            }

            // Apply final norm
            let normed_hidden = self.norm.forward(&current_hidden)?;

            // Predict next codebook
            // Python: logits = self.lm_head[generation_steps](hidden_states)
            let logits = self.lm_heads[gen_step].forward(&normed_hidden)?;

            predicted_code = self.sample_from_logits(&logits, &mut sampler, batch_size)?;
            all_codes.push(predicted_code.clone());
            debug!(
                "CodePredictor: codebook {} = {:?}",
                gen_step + 1,
                predicted_code.to_vec2::<u32>()?
            );

            position_offset += 1;
        }

        // Stack all codebooks: [batch, 1, num_groups]
        let stacked: Vec<Tensor> = all_codes
            .iter()
            .map(|t| t.unsqueeze(2))
            .collect::<Result<Vec<_>>>()?;

        Tensor::cat(&stacked, 2)
    }

    /// Sample tokens from logits.
    fn sample_from_logits(
        &self,
        logits: &Tensor,
        sampler: &mut Sampler,
        batch_size: usize,
    ) -> Result<Tensor> {
        let mut codes = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let pos_logits = logits.i((b, 0, ..))?;
            let logits_vec: Vec<f32> = pos_logits.to_vec1()?;
            let token = sampler.sample(&logits_vec);
            codes.push(token);
        }

        Tensor::new(codes.as_slice(), &self.device)?.unsqueeze(1)
    }

    // Note: The old `predict` and `predict_frame` methods have been removed.
    // CodePredictor now requires proper hidden states and zeroth embeddings from Talker.
    // Use `predict_from_hidden` with correct arguments instead.
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
    fn test_code_predictor_get_codec_embedding() {
        let device = Device::Cpu;
        let config = test_config();

        let predictor = CodePredictor::new_random(config.clone(), &device).unwrap();

        // Test getting embedding for codebook 0 (residual codebook 1)
        let token_ids = Tensor::new(&[10u32], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap(); // [1, 1]
        let emb = predictor.get_codec_embedding(0, &token_ids).unwrap();

        assert_eq!(emb.dims(), &[1, 1, config.hidden_size]);
    }
}
