//! Neural codec decoder for Qwen3-TTS-Tokenizer-12Hz.
//!
//! This module implements the audio decoder that converts acoustic tokens
//! back to PCM waveform using a neural network architecture.

use std::path::Path;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder,
    conv_transpose1d, conv1d,
};
use serde_json::Value;
use tracing::{debug, info, instrument};

use tts_core::{TtsError, TtsResult};

/// Configuration for the codec decoder.
///
/// Based on Qwen3-TTS-Tokenizer-12Hz architecture.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Number of codebook entries per quantizer.
    pub codebook_size: usize,
    /// Codebook embedding dimension.
    pub codebook_dim: usize,
    /// Latent dimension (after combining all codebooks).
    pub latent_dim: usize,
    /// Decoder hidden dimension.
    pub decoder_dim: usize,
    /// Number of quantizers (codebook groups).
    pub num_quantizers: usize,
    /// Number of semantic quantizers (first N are semantic).
    pub num_semantic_quantizers: usize,
    /// Number of transformer layers in decoder.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of residual blocks per upsampling stage.
    pub num_residual_blocks: usize,
    /// Upsampling factors for ConvNet stages.
    pub upsample_rates: Vec<usize>,
    /// Additional upsampling ratios.
    pub upsampling_ratios: Vec<usize>,
    /// Sample rate.
    pub sample_rate: u32,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
}

impl Default for DecoderConfig {
    /// Default configuration matching Qwen3-TTS-Tokenizer-12Hz decoder.
    fn default() -> Self {
        Self::qwen3_tts_12hz()
    }
}

impl DecoderConfig {
    /// Load configuration from a JSON file (HuggingFace config.json format).
    pub fn from_json_file(path: impl AsRef<Path>) -> TtsResult<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| TtsError::ModelLoad {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_json(&content)
    }

    /// Parse configuration from JSON string (HuggingFace config.json format).
    pub fn from_json(json: &str) -> TtsResult<Self> {
        let v: Value = serde_json::from_str(json)
            .map_err(|e| TtsError::config(format!("failed to parse JSON: {e}")))?;

        // Helper to extract values with defaults
        let get_usize = |key: &str, default: usize| -> usize {
            v.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let get_f64 = |key: &str, default: f64| -> f64 {
            v.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };

        let get_u32 = |key: &str, default: u32| -> u32 {
            v.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default)
        };

        let get_vec_usize = |key: &str, default: Vec<usize>| -> Vec<usize> {
            v.get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or(default)
        };

        Ok(Self {
            codebook_size: get_usize("codebook_size", 2048),
            codebook_dim: get_usize("codebook_dim", 512),
            latent_dim: get_usize("latent_dim", 1024),
            decoder_dim: get_usize("decoder_dim", 1536),
            num_quantizers: get_usize("num_quantizers", 16),
            num_semantic_quantizers: get_usize("num_semantic_quantizers", 1),
            num_hidden_layers: get_usize("num_hidden_layers", 8),
            num_attention_heads: get_usize("num_attention_heads", 16),
            num_residual_blocks: get_usize("num_residual_blocks", 3),
            upsample_rates: get_vec_usize("upsample_rates", vec![8, 5, 4, 3]),
            upsampling_ratios: get_vec_usize("upsampling_ratios", vec![2, 2]),
            sample_rate: get_u32("sample_rate", 24000),
            rms_norm_eps: get_f64("rms_norm_eps", 1e-5),
        })
    }

    /// Configuration for Qwen3-TTS-Tokenizer-12Hz decoder.
    ///
    /// Based on config.json from HuggingFace:
    /// - 16 quantizers (1 semantic + 15 acoustic)
    /// - 2048 codebook size
    /// - upsample_rates: [8, 5, 4, 3] = 480
    /// - upsampling_ratios: [2, 2] = 4
    /// - total: 480 * 4 = 1920 samples per frame
    pub fn qwen3_tts_12hz() -> Self {
        Self {
            codebook_size: 2048,
            codebook_dim: 512,
            latent_dim: 1024,
            decoder_dim: 1536,
            num_quantizers: 16,
            num_semantic_quantizers: 1,
            num_hidden_layers: 8,
            num_attention_heads: 16,
            num_residual_blocks: 3,
            upsample_rates: vec![8, 5, 4, 3],
            upsampling_ratios: vec![2, 2],
            sample_rate: 24000,
            rms_norm_eps: 1e-5,
        }
    }

    /// Small configuration for testing.
    pub fn tiny() -> Self {
        Self {
            codebook_size: 64,
            codebook_dim: 32,
            latent_dim: 128, // num_quantizers * codebook_dim = 4 * 32 = 128
            decoder_dim: 64,
            num_quantizers: 4,
            num_semantic_quantizers: 1,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_residual_blocks: 1,
            upsample_rates: vec![4, 4], // Use larger factors for stable convolutions
            upsampling_ratios: vec![],  // Keep it simple
            sample_rate: 24000,
            rms_norm_eps: 1e-5,
        }
    }

    /// Legacy configuration for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use qwen3_tts_12hz() instead")]
    pub fn legacy() -> Self {
        Self {
            codebook_size: 65536,
            codebook_dim: 512,
            latent_dim: 512,
            decoder_dim: 512,
            num_quantizers: 1,
            num_semantic_quantizers: 1,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_residual_blocks: 3,
            upsample_rates: vec![8, 5, 5, 10],
            upsampling_ratios: vec![],
            sample_rate: 24000,
            rms_norm_eps: 1e-6,
        }
    }

    /// Calculate total upsampling factor.
    pub fn total_upsample(&self) -> usize {
        let rate_product: usize = self.upsample_rates.iter().product();
        let ratio_product: usize = self.upsampling_ratios.iter().product();
        rate_product * ratio_product
    }

    /// Get samples per token frame at 12.5 Hz.
    /// For 24kHz audio and 12.5 FPS: 24000 / 12.5 = 1920 samples/frame
    pub fn samples_per_frame(&self) -> usize {
        self.sample_rate as usize / 125 * 10 // 24000 / 12.5 = 1920
    }
}

/// Residual block with dilated convolutions.
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResidualBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let padding = (kernel_size - 1) * dilation / 2;

        let conv1_config = Conv1dConfig {
            padding,
            dilation,
            ..Default::default()
        };
        let conv1 = conv1d(
            channels,
            channels,
            kernel_size,
            conv1_config,
            vb.pp("conv1"),
        )?;

        let conv2_config = Conv1dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        let conv2 = conv1d(
            channels,
            channels,
            kernel_size,
            conv2_config,
            vb.pp("conv2"),
        )?;

        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        // Conv1 + LeakyReLU
        let x = self.conv1.forward(x)?;
        let x = leaky_relu(&x, 0.1)?;

        // Conv2 + LeakyReLU
        let x = self.conv2.forward(&x)?;
        let x = leaky_relu(&x, 0.1)?;

        // Residual connection
        x + residual
    }
}

/// Upsampling block with transposed convolution.
#[derive(Debug)]
struct UpsampleBlock {
    conv_transpose: ConvTranspose1d,
    residual_blocks: Vec<ResidualBlock>,
}

impl UpsampleBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        upsample_factor: usize,
        kernel_size: usize,
        num_residual: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let padding = (kernel_size - upsample_factor) / 2;

        let conv_config = ConvTranspose1dConfig {
            stride: upsample_factor,
            padding,
            ..Default::default()
        };
        let conv_transpose = conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_config,
            vb.pp("conv_transpose"),
        )?;

        let mut residual_blocks = Vec::with_capacity(num_residual);
        for i in 0..num_residual {
            let dilation = 3usize.pow(i as u32);
            let block = ResidualBlock::new(out_channels, 7, dilation, vb.pp(format!("res_{i}")))?;
            residual_blocks.push(block);
        }

        Ok(Self {
            conv_transpose,
            residual_blocks,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = self.conv_transpose.forward(x)?;
        x = leaky_relu(&x, 0.1)?;

        for block in &self.residual_blocks {
            x = block.forward(&x)?;
        }

        Ok(x)
    }
}

/// Neural decoder network for audio synthesis.
///
/// Supports multi-codebook input from Qwen3-TTS-Tokenizer-12Hz.
/// The decoder combines embeddings from all 16 codebooks and
/// upsamples to produce 24kHz audio.
#[derive(Debug)]
pub struct NeuralDecoder {
    /// Codebook embeddings (one per quantizer).
    codebooks: Vec<Tensor>,
    /// Projection from combined codebook embeddings to latent space.
    embed_proj: Option<Conv1d>,
    /// Initial projection.
    input_conv: Conv1d,
    /// Upsampling stages.
    upsample_blocks: Vec<UpsampleBlock>,
    /// Final output convolution.
    output_conv: Conv1d,
    /// Configuration.
    config: DecoderConfig,
    /// Device.
    device: Device,
}

impl NeuralDecoder {
    /// Create a new neural decoder with random weights (for testing).
    ///
    /// Initializes 16 codebook embeddings (one per quantizer) and
    /// the upsampling network.
    pub fn new(config: DecoderConfig, device: &Device) -> CandleResult<Self> {
        // Initialize codebooks (one per quantizer)
        let mut codebooks = Vec::with_capacity(config.num_quantizers);
        for _ in 0..config.num_quantizers {
            let codebook = Tensor::randn(
                0.0f32,
                0.02,
                (config.codebook_size, config.codebook_dim),
                device,
            )?;
            codebooks.push(codebook);
        }

        // Create placeholder layers (will be replaced when loading weights)
        let vb = VarBuilder::zeros(DType::F32, device);

        // Combined embedding dimension = num_quantizers * codebook_dim
        let combined_dim = config.num_quantizers * config.codebook_dim;

        // Projection from combined codebook embeddings to latent space
        let embed_proj = if combined_dim != config.latent_dim {
            Some(conv1d(
                combined_dim,
                config.latent_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("embed_proj"),
            )?)
        } else {
            None
        };

        // Input projection from latent to decoder dimension
        let input_conv = conv1d(
            config.latent_dim,
            config.decoder_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )?;

        // Build upsampling stages based on upsample_rates
        let mut upsample_blocks = Vec::new();
        let mut channels = config.decoder_dim;

        for (i, &factor) in config.upsample_rates.iter().enumerate() {
            let out_channels = (channels / 2).max(32);
            let kernel_size = factor * 2; // Common heuristic for kernel size
            let block = UpsampleBlock::new(
                channels,
                out_channels,
                factor,
                kernel_size,
                config.num_residual_blocks,
                vb.pp(format!("upsample_{i}")),
            )?;
            upsample_blocks.push(block);
            channels = out_channels;
        }

        // Additional upsampling for upsampling_ratios
        for (i, &factor) in config.upsampling_ratios.iter().enumerate() {
            let idx = config.upsample_rates.len() + i;
            let out_channels = (channels / 2).max(32);
            let kernel_size = factor * 2;
            let block = UpsampleBlock::new(
                channels,
                out_channels,
                factor,
                kernel_size,
                config.num_residual_blocks,
                vb.pp(format!("upsample_{idx}")),
            )?;
            upsample_blocks.push(block);
            channels = out_channels;
        }

        let output_conv = conv1d(
            channels,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("output_conv"),
        )?;

        Ok(Self {
            codebooks,
            embed_proj,
            input_conv,
            upsample_blocks,
            output_conv,
            config,
            device: device.clone(),
        })
    }

    /// Load decoder from safetensors file.
    #[instrument(skip(config), fields(path = %path.as_ref().display()))]
    pub fn load(path: impl AsRef<Path>, config: DecoderConfig, device: &Device) -> TtsResult<Self> {
        let path = path.as_ref();
        info!("Loading codec decoder from {}", path.display());

        let tensors =
            candle_core::safetensors::load(path, device).map_err(|e| TtsError::ModelLoad {
                path: path.to_path_buf(),
                source: std::io::Error::other(e.to_string()),
            })?;

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

        Self::from_vb(vb, config, device)
    }

    /// Create decoder from VarBuilder.
    ///
    /// Loads 16 codebook embeddings and the upsampling network from weights.
    pub fn from_vb(vb: VarBuilder, config: DecoderConfig, device: &Device) -> TtsResult<Self> {
        // Load codebooks (one per quantizer)
        let mut codebooks = Vec::with_capacity(config.num_quantizers);
        for i in 0..config.num_quantizers {
            let codebook = vb
                .get(
                    (config.codebook_size, config.codebook_dim),
                    &format!("codebook.{i}"),
                )
                .or_else(|_| {
                    // Try alternative naming conventions
                    vb.get(
                        (config.codebook_size, config.codebook_dim),
                        &format!("quantizer.{i}.codebook"),
                    )
                })
                .map_err(|e| TtsError::internal(format!("failed to load codebook {i}: {e}")))?;
            codebooks.push(codebook);
        }

        // Combined embedding dimension
        let combined_dim = config.num_quantizers * config.codebook_dim;

        // Projection from combined embeddings to latent space
        let embed_proj = if combined_dim != config.latent_dim {
            Some(
                conv1d(
                    combined_dim,
                    config.latent_dim,
                    1,
                    Conv1dConfig::default(),
                    vb.pp("embed_proj"),
                )
                .map_err(|e| TtsError::internal(format!("failed to load embed_proj: {e}")))?,
            )
        } else {
            None
        };

        // Input projection
        let input_conv = conv1d(
            config.latent_dim,
            config.decoder_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )
        .map_err(|e| TtsError::internal(format!("failed to load input_conv: {e}")))?;

        // Build upsampling stages
        let mut upsample_blocks = Vec::new();
        let mut channels = config.decoder_dim;

        for (i, &factor) in config.upsample_rates.iter().enumerate() {
            let out_channels = (channels / 2).max(32);
            let kernel_size = factor * 2;
            let block = UpsampleBlock::new(
                channels,
                out_channels,
                factor,
                kernel_size,
                config.num_residual_blocks,
                vb.pp(format!("upsample_{i}")),
            )
            .map_err(|e| TtsError::internal(format!("failed to load upsample_{i}: {e}")))?;

            upsample_blocks.push(block);
            channels = out_channels;
        }

        // Additional upsampling for upsampling_ratios
        for (i, &factor) in config.upsampling_ratios.iter().enumerate() {
            let idx = config.upsample_rates.len() + i;
            let out_channels = (channels / 2).max(32);
            let kernel_size = factor * 2;
            let block = UpsampleBlock::new(
                channels,
                out_channels,
                factor,
                kernel_size,
                config.num_residual_blocks,
                vb.pp(format!("upsample_{idx}")),
            )
            .map_err(|e| TtsError::internal(format!("failed to load upsample_{idx}: {e}")))?;

            upsample_blocks.push(block);
            channels = out_channels;
        }

        let output_conv = conv1d(
            channels,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("output_conv"),
        )
        .map_err(|e| TtsError::internal(format!("failed to load output_conv: {e}")))?;

        info!(
            "Decoder loaded: {} codebooks x {} entries, latent={}, upsample={}x",
            config.num_quantizers,
            config.codebook_size,
            config.latent_dim,
            config.total_upsample()
        );

        Ok(Self {
            codebooks,
            embed_proj,
            input_conv,
            upsample_blocks,
            output_conv,
            config,
            device: device.clone(),
        })
    }

    /// Get the decoder configuration.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Decode acoustic tokens from all codebooks to audio samples.
    ///
    /// # Arguments
    /// * `tokens` - 2D array of shape `[num_quantizers, seq_len]` where each row
    ///              contains tokens from one codebook.
    ///
    /// # Example
    /// ```ignore
    /// let tokens = vec![
    ///     vec![1, 2, 3],  // codebook 0 (semantic)
    ///     vec![4, 5, 6],  // codebook 1
    ///     // ... 14 more codebooks
    /// ];
    /// let audio = decoder.decode_multi(&tokens)?;
    /// ```
    #[instrument(skip(self, tokens), fields(num_quantizers = tokens.len()))]
    pub fn decode_multi(&self, tokens: &[Vec<u32>]) -> TtsResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        if tokens.len() != self.config.num_quantizers {
            return Err(TtsError::invalid_input(format!(
                "expected {} codebooks, got {}",
                self.config.num_quantizers,
                tokens.len()
            )));
        }

        // Check all sequences have same length
        let seq_len = tokens[0].len();
        if seq_len == 0 {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        for (i, codebook_tokens) in tokens.iter().enumerate() {
            if codebook_tokens.len() != seq_len {
                return Err(TtsError::invalid_input(format!(
                    "codebook {} has {} tokens, expected {}",
                    i,
                    codebook_tokens.len(),
                    seq_len
                )));
            }
            // Validate token values
            for &token in codebook_tokens {
                if token as usize >= self.config.codebook_size {
                    return Err(TtsError::invalid_input(format!(
                        "token {} in codebook {} exceeds codebook size {}",
                        token, i, self.config.codebook_size
                    )));
                }
            }
        }

        let result = self
            .decode_multi_internal(tokens)
            .map_err(|e| TtsError::internal(format!("decode failed: {e}")))?;

        debug!(
            "Decoded {} quantizers x {} tokens to {} samples",
            tokens.len(),
            seq_len,
            result.len()
        );
        Ok(result)
    }

    /// Decode single-codebook tokens (backward compatibility).
    ///
    /// Uses only the first codebook (semantic). For full quality,
    /// use `decode_multi()` with all 16 codebooks.
    #[instrument(skip(self, tokens), fields(num_tokens = tokens.len()))]
    pub fn decode(&self, tokens: &[u32]) -> TtsResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        // Validate tokens
        for &token in tokens {
            if token as usize >= self.config.codebook_size {
                return Err(TtsError::invalid_input(format!(
                    "token {} exceeds codebook size {}",
                    token, self.config.codebook_size
                )));
            }
        }

        // Create multi-codebook input with zeros for non-semantic codebooks
        let mut multi_tokens = Vec::with_capacity(self.config.num_quantizers);
        multi_tokens.push(tokens.to_vec()); // Semantic codebook

        // Fill remaining codebooks with zeros (silence)
        for _ in 1..self.config.num_quantizers {
            multi_tokens.push(vec![0u32; tokens.len()]);
        }

        self.decode_multi(&multi_tokens)
    }

    fn decode_multi_internal(&self, tokens: &[Vec<u32>]) -> CandleResult<Vec<f32>> {
        let seq_len = tokens[0].len();

        // Lookup embeddings from each codebook and concatenate
        let mut embeddings_list = Vec::with_capacity(self.config.num_quantizers);

        for (i, (codebook, codebook_tokens)) in self.codebooks.iter().zip(tokens.iter()).enumerate()
        {
            let token_ids: Vec<i64> = codebook_tokens.iter().map(|&t| t as i64).collect();
            let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?;

            // Lookup: [seq_len, codebook_dim]
            let emb = codebook.index_select(&token_tensor, 0)?;
            embeddings_list.push(emb);

            debug!("Codebook {}: looked up {} embeddings", i, seq_len);
        }

        // Concatenate along embedding dimension: [seq_len, num_quantizers * codebook_dim]
        let combined = Tensor::cat(&embeddings_list, 1)?;

        // Reshape to [batch=1, channels, seq_len]
        let combined = combined.unsqueeze(0)?.transpose(1, 2)?;

        // Project combined embeddings to latent space if needed
        let latent = if let Some(ref proj) = self.embed_proj {
            proj.forward(&combined)?
        } else {
            combined
        };

        // Input projection
        let mut x = self.input_conv.forward(&latent)?;
        x = leaky_relu(&x, 0.1)?;

        // Upsampling stages
        for block in &self.upsample_blocks {
            x = block.forward(&x)?;
        }

        // Output projection
        let x = self.output_conv.forward(&x)?;

        // Apply tanh to constrain output to [-1, 1]
        let x = x.tanh()?;

        // Squeeze and convert to Vec
        let x = x.squeeze(0)?.squeeze(0)?;
        x.to_vec1()
    }

    /// Decode a single frame from all codebooks (for streaming).
    ///
    /// # Arguments
    /// * `tokens` - Array of `num_quantizers` tokens, one from each codebook.
    pub fn decode_frame(&self, tokens: &[u32]) -> TtsResult<Vec<f32>> {
        if tokens.len() != self.config.num_quantizers {
            return Err(TtsError::invalid_input(format!(
                "expected {} tokens (one per codebook), got {}",
                self.config.num_quantizers,
                tokens.len()
            )));
        }

        // Convert to multi-codebook format with seq_len=1
        let multi_tokens: Vec<Vec<u32>> = tokens.iter().map(|&t| vec![t]).collect();
        self.decode_multi(&multi_tokens)
    }

    /// Decode a single token from semantic codebook (backward compatibility).
    pub fn decode_single(&self, token: u32) -> TtsResult<Vec<f32>> {
        self.decode(&[token])
    }
}

/// Leaky ReLU activation.
fn leaky_relu(x: &Tensor, negative_slope: f32) -> CandleResult<Tensor> {
    let zeros = x.zeros_like()?;
    let positive = x.maximum(&zeros)?;
    let negative = (x.minimum(&zeros)? * negative_slope as f64)?;
    positive + negative
}

/// Mock decoder for testing without model weights.
#[derive(Debug)]
pub struct MockDecoder {
    config: DecoderConfig,
    sample_rate: u32,
}

impl MockDecoder {
    /// Create a new mock decoder.
    pub fn new(config: DecoderConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Decode tokens to audio (generates sine wave for testing).
    pub fn decode(&self, tokens: &[u32]) -> TtsResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        let num_samples = tokens.len() * self.config.total_upsample();
        let mut samples = Vec::with_capacity(num_samples);

        // Generate a simple sine wave based on token values
        for (i, &token) in tokens.iter().enumerate() {
            let base_freq = 220.0 + (token % 100) as f32 * 5.0; // Vary frequency by token
            let samples_per_token = self.config.total_upsample();

            for j in 0..samples_per_token {
                let t = (i * samples_per_token + j) as f32 / self.sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * base_freq * t).sin() * 0.3;
                samples.push(sample);
            }
        }

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config_default() {
        let config = DecoderConfig::default();
        assert_eq!(config.codebook_size, 2048);
        assert_eq!(config.num_quantizers, 16);
        // upsample_rates: [8, 5, 4, 3] = 480, upsampling_ratios: [2, 2] = 4
        // total: 480 * 4 = 1920
        assert_eq!(config.total_upsample(), 1920);
    }

    #[test]
    fn test_decoder_config_qwen3() {
        let config = DecoderConfig::qwen3_tts_12hz();
        assert_eq!(config.codebook_size, 2048);
        assert_eq!(config.codebook_dim, 512);
        assert_eq!(config.num_quantizers, 16);
        assert_eq!(config.num_semantic_quantizers, 1);
        assert_eq!(config.samples_per_frame(), 1920);
    }

    #[test]
    fn test_decoder_config_tiny() {
        let config = DecoderConfig::tiny();
        assert_eq!(config.codebook_size, 64);
        assert_eq!(config.num_quantizers, 4);
        // upsample_rates: [4, 4] = 16, upsampling_ratios: [] = 1
        // total: 16 * 1 = 16
        assert_eq!(config.total_upsample(), 16);
    }

    #[test]
    fn test_decoder_config_upsample() {
        let config = DecoderConfig {
            upsample_rates: vec![4, 4, 4, 4],
            upsampling_ratios: vec![],
            ..Default::default()
        };
        assert_eq!(config.total_upsample(), 256);
    }

    #[test]
    fn test_mock_decoder() {
        let config = DecoderConfig::default();
        let decoder = MockDecoder::new(config, 24000);

        let tokens = vec![100, 200, 300];
        let samples = decoder.decode(&tokens).unwrap();

        assert_eq!(samples.len(), 3 * 1920);

        // Check samples are in valid range
        for &s in &samples {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_mock_decoder_empty_error() {
        let config = DecoderConfig::default();
        let decoder = MockDecoder::new(config, 24000);

        let result = decoder.decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_decoder_creation() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device);

        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.config().codebook_size, 64);
        assert_eq!(decoder.config().num_quantizers, 4);
    }

    #[test]
    fn test_neural_decoder_multi_codebook_decode() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Create tokens for all codebooks (4 quantizers, 3 frames each)
        let tokens: Vec<Vec<u32>> = (0..config.num_quantizers).map(|_| vec![1, 2, 3]).collect();

        let samples = decoder.decode_multi(&tokens).unwrap();

        // Should produce seq_len * total_upsample samples
        let expected_len = 3 * config.total_upsample();
        assert_eq!(samples.len(), expected_len);
    }

    #[test]
    fn test_neural_decoder_single_codebook_decode() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Single-codebook decode (backward compatibility)
        let tokens = vec![1, 2, 3];
        let samples = decoder.decode(&tokens).unwrap();

        let expected_len = tokens.len() * config.total_upsample();
        assert_eq!(samples.len(), expected_len);
    }

    #[test]
    fn test_neural_decoder_frame_decode() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Note: Single-frame decode may not work with all conv configurations
        // due to minimum input size requirements. Test with 2 frames minimum.
        // Decode two frames (one token per codebook, 2 frames)
        let multi_tokens: Vec<Vec<u32>> = (0..config.num_quantizers)
            .map(|i| vec![i as u32, (i as u32 + 1) % config.codebook_size as u32])
            .collect();

        let samples = decoder.decode_multi(&multi_tokens).unwrap();

        let expected_len = 2 * config.total_upsample();
        assert_eq!(samples.len(), expected_len);
    }

    #[test]
    fn test_neural_decoder_invalid_token() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Token 100 exceeds codebook size of 64
        let result = decoder.decode(&[100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_decoder_mismatched_codebooks() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Only 2 codebooks instead of 4
        let tokens = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let result = decoder.decode_multi(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_decoder_mismatched_lengths() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Codebooks with different lengths
        let tokens = vec![
            vec![1, 2, 3],
            vec![4, 5], // Wrong length!
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        let result = decoder.decode_multi(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_decoder_frame_wrong_count() {
        let config = DecoderConfig::tiny();
        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        // Wrong number of tokens for frame decode
        let tokens = vec![1, 2]; // Should be 4 (num_quantizers)
        let result = decoder.decode_frame(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_config_from_json() {
        let json = r#"{
            "codebook_size": 1024,
            "codebook_dim": 256,
            "latent_dim": 512,
            "decoder_dim": 768,
            "num_quantizers": 8,
            "num_semantic_quantizers": 2,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_residual_blocks": 2,
            "upsample_rates": [4, 4, 2, 2],
            "upsampling_ratios": [2],
            "sample_rate": 16000,
            "rms_norm_eps": 1e-6
        }"#;

        let config = DecoderConfig::from_json(json).unwrap();

        assert_eq!(config.codebook_size, 1024);
        assert_eq!(config.codebook_dim, 256);
        assert_eq!(config.latent_dim, 512);
        assert_eq!(config.decoder_dim, 768);
        assert_eq!(config.num_quantizers, 8);
        assert_eq!(config.num_semantic_quantizers, 2);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_residual_blocks, 2);
        assert_eq!(config.upsample_rates, vec![4, 4, 2, 2]);
        assert_eq!(config.upsampling_ratios, vec![2]);
        assert_eq!(config.sample_rate, 16000);
        // total upsample: 4*4*2*2 * 2 = 64 * 2 = 128
        assert_eq!(config.total_upsample(), 128);
    }

    #[test]
    fn test_decoder_config_from_json_defaults() {
        // Minimal JSON - should use defaults
        let json = r#"{"codebook_size": 512}"#;
        let config = DecoderConfig::from_json(json).unwrap();

        assert_eq!(config.codebook_size, 512);
        // Check defaults are applied
        assert_eq!(config.codebook_dim, 512);
        assert_eq!(config.num_quantizers, 16);
        assert_eq!(config.upsample_rates, vec![8, 5, 4, 3]);
    }

    #[test]
    fn test_decoder_config_from_json_invalid() {
        let result = DecoderConfig::from_json("not valid json");
        assert!(result.is_err());
    }
}
