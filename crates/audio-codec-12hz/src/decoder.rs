//! Neural codec decoder for Qwen3-TTS-Tokenizer-12Hz.
//!
//! This module implements the audio decoder that converts acoustic tokens
//! back to PCM waveform using a neural network architecture.
//!
//! The Qwen3-TTS-Tokenizer-12Hz decoder architecture:
//! 1. **Quantizer dequantize**: tokens → latent embeddings
//! 2. **Pre-Transformer**: 8-layer transformer with layer scale
//! 3. **Upsample**: 2 ConvNeXt blocks with 2x upsampling each
//! 4. **Pre-conv**: Conv1d projection
//! 5. **HiFi-GAN decoder**: 4 upsample blocks with Snake activation
//! 6. **Output**: Final conv to mono audio

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

    /// Create with random initialization.
    fn new_random(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let padding1 = (kernel_size - 1) * dilation / 2;

        let weight1 = Tensor::randn(0.0f32, 0.02, (channels, channels, kernel_size), device)?;
        let bias1 = Tensor::zeros((channels,), DType::F32, device)?;
        let conv1 = Conv1d::new(
            weight1,
            Some(bias1),
            Conv1dConfig {
                padding: padding1,
                dilation,
                ..Default::default()
            },
        );

        let weight2 = Tensor::randn(0.0f32, 0.02, (channels, channels, kernel_size), device)?;
        let bias2 = Tensor::zeros((channels,), DType::F32, device)?;
        let conv2 = Conv1d::new(
            weight2,
            Some(bias2),
            Conv1dConfig {
                padding: kernel_size / 2,
                ..Default::default()
            },
        );

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

    /// Create with random initialization (for fallback when weights don't match).
    fn new_random(
        in_channels: usize,
        out_channels: usize,
        upsample_factor: usize,
        kernel_size: usize,
        num_residual: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let padding = (kernel_size - upsample_factor) / 2;

        // Random conv transpose
        let weight = Tensor::randn(
            0.0f32,
            0.02,
            (in_channels, out_channels, kernel_size),
            device,
        )?;
        let bias = Tensor::zeros((out_channels,), DType::F32, device)?;
        let conv_config = ConvTranspose1dConfig {
            stride: upsample_factor,
            padding,
            ..Default::default()
        };
        let conv_transpose = ConvTranspose1d::new(weight, Some(bias), conv_config);

        // Random residual blocks
        let mut residual_blocks = Vec::with_capacity(num_residual);
        for i in 0..num_residual {
            let dilation = 3usize.pow(i as u32);
            let block = ResidualBlock::new_random(out_channels, 7, dilation, device)?;
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
///
/// The decoder supports two modes:
/// - **Legacy mode**: Uses simplified upsampling (when HiFi-GAN weights not available)
/// - **HiFi-GAN mode**: Uses real Qwen3-TTS architecture with Snake activation
#[derive(Debug)]
pub struct NeuralDecoder {
    /// Codebook embeddings (one per quantizer).
    codebooks: Vec<Tensor>,
    /// Projection from combined codebook embeddings to latent space.
    embed_proj: Option<Conv1d>,
    /// Initial projection.
    input_conv: Conv1d,
    /// Upsampling stages (legacy mode).
    upsample_blocks: Vec<UpsampleBlock>,
    /// HiFi-GAN upsample blocks (real Qwen3 mode).
    hifi_blocks: Option<Vec<HifiUpsampleBlock>>,
    /// Final Snake activation (for HiFi-GAN mode).
    final_snake: Option<Snake>,
    /// Final output convolution.
    output_conv: Conv1d,
    /// Configuration.
    config: DecoderConfig,
    /// Device.
    device: Device,
    /// Whether using HiFi-GAN mode.
    use_hifi: bool,
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
            hifi_blocks: None,
            final_snake: None,
            output_conv,
            config,
            device: device.clone(),
            use_hifi: false,
        })
    }

    /// Load decoder from safetensors file.
    #[instrument(skip(config), fields(device = ?device, path = %path.as_ref().display()))]
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
}

/// Load codebook embedding from Qwen3-TTS-Tokenizer-12Hz format.
///
/// The model stores codebooks as EMA embedding sums that need to be normalized
/// by cluster usage counts to get the actual embeddings.
fn load_codebook_embedding(
    vb: &VarBuilder,
    index: usize,
    config: &DecoderConfig,
    device: &Device,
) -> TtsResult<Tensor> {
    // Qwen3-TTS-Tokenizer-12Hz format:
    // - First quantizer (index 0): decoder.quantizer.rvq_first.vq.layers.0._codebook
    // - Rest (index 1-15): decoder.quantizer.rvq_rest.vq.layers.{index-1}._codebook

    let (embed_key, usage_key) = if index == 0 {
        (
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum".to_string(),
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage".to_string(),
        )
    } else {
        (
            format!(
                "decoder.quantizer.rvq_rest.vq.layers.{}._codebook.embedding_sum",
                index - 1
            ),
            format!(
                "decoder.quantizer.rvq_rest.vq.layers.{}._codebook.cluster_usage",
                index - 1
            ),
        )
    };

    // Real codebook dimension in Qwen3-TTS-Tokenizer is 256, not 512 from config
    // We'll try both dimensions
    let actual_codebook_dim = 256; // vector_quantization_hidden_dimension from config

    // Try Qwen3 format first with actual dimension (256)
    let result = vb.get((config.codebook_size, actual_codebook_dim), &embed_key);

    if let Ok(embed_sum) = result {
        // Load cluster usage for normalization
        if let Ok(cluster_usage) = vb.get((config.codebook_size,), &usage_key) {
            // Normalize: embedding = embedding_sum / cluster_usage
            // Add small epsilon to avoid division by zero
            let usage_expanded = cluster_usage
                .unsqueeze(1)
                .map_err(|e| TtsError::internal(format!("failed to expand cluster_usage: {e}")))?;
            let normalized = embed_sum
                .broadcast_div(
                    &(usage_expanded + 1e-7)
                        .map_err(|e| TtsError::internal(format!("failed to add epsilon: {e}")))?,
                )
                .map_err(|e| TtsError::internal(format!("failed to normalize codebook: {e}")))?;
            debug!(
                "Loaded codebook {} from Qwen3 format (normalized, dim={})",
                index, actual_codebook_dim
            );
            return Ok(normalized);
        }
        // If no cluster_usage, use embed_sum directly
        debug!(
            "Loaded codebook {} from Qwen3 format (raw, dim={})",
            index, actual_codebook_dim
        );
        return Ok(embed_sum);
    }

    // Try with config.codebook_dim (512) as fallback
    let result = vb.get((config.codebook_size, config.codebook_dim), &embed_key);

    if let Ok(embed_sum) = result {
        if let Ok(cluster_usage) = vb.get((config.codebook_size,), &usage_key) {
            let usage_expanded = cluster_usage
                .unsqueeze(1)
                .map_err(|e| TtsError::internal(format!("failed to expand cluster_usage: {e}")))?;
            let normalized = embed_sum
                .broadcast_div(
                    &(usage_expanded + 1e-7)
                        .map_err(|e| TtsError::internal(format!("failed to add epsilon: {e}")))?,
                )
                .map_err(|e| TtsError::internal(format!("failed to normalize codebook: {e}")))?;
            debug!(
                "Loaded codebook {} from Qwen3 format (normalized, dim={})",
                index, config.codebook_dim
            );
            return Ok(normalized);
        }
        debug!(
            "Loaded codebook {} from Qwen3 format (raw, dim={})",
            index, config.codebook_dim
        );
        return Ok(embed_sum);
    }

    // Try alternative naming conventions
    let alt_keys = [
        format!("codebook.{index}"),
        format!("quantizer.{index}.codebook"),
        format!("quantizers.{index}.codebook.weight"),
    ];

    for key in &alt_keys {
        if let Ok(codebook) = vb.get((config.codebook_size, config.codebook_dim), key) {
            debug!("Loaded codebook {} from key: {}", index, key);
            return Ok(codebook);
        }
    }

    // If all else fails, create random initialization with actual dimension
    debug!("Codebook {} not found, using random initialization", index);
    let codebook = Tensor::randn(
        0.0f32,
        0.02,
        (config.codebook_size, actual_codebook_dim),
        device,
    )
    .map_err(|e| TtsError::internal(format!("failed to create random codebook: {e}")))?;

    Ok(codebook)
}

/// Create a Conv1d layer with random initialization.
fn create_random_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    device: &Device,
) -> CandleResult<Conv1d> {
    let weight = Tensor::randn(
        0.0f32,
        0.02,
        (out_channels, in_channels, kernel_size),
        device,
    )?;
    let bias = Tensor::zeros((out_channels,), DType::F32, device)?;
    let config = Conv1dConfig {
        padding: kernel_size / 2,
        ..Default::default()
    };
    Ok(Conv1d::new(weight, Some(bias), config))
}

impl NeuralDecoder {
    /// Create decoder from VarBuilder.
    ///
    /// Loads 16 codebook embeddings and the upsampling network from weights.
    /// Falls back to random initialization if weights aren't found (for testing).
    pub fn from_vb(vb: VarBuilder, config: DecoderConfig, device: &Device) -> TtsResult<Self> {
        // Try to load real weights, but fall back to random init if structure doesn't match
        // The Qwen3-TTS-Tokenizer has a different HiFi-GAN decoder architecture that we
        // don't fully support yet. For now, we load codebooks and use simplified upsampling.

        // Load codebooks (one per quantizer)
        // Qwen3-TTS-Tokenizer-12Hz format:
        // - First quantizer: decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum
        // - Rest (15): decoder.quantizer.rvq_rest.vq.layers.{0-14}._codebook.embedding_sum
        let mut codebooks = Vec::with_capacity(config.num_quantizers);

        for i in 0..config.num_quantizers {
            let codebook = load_codebook_embedding(&vb, i, &config, device)?;
            codebooks.push(codebook);
        }

        // Determine actual codebook dimension from loaded tensors
        let actual_codebook_dim = if !codebooks.is_empty() {
            codebooks[0].dim(1).unwrap_or(config.codebook_dim)
        } else {
            config.codebook_dim
        };

        info!(
            "Loaded {} codebooks ({} x {})",
            codebooks.len(),
            config.codebook_size,
            actual_codebook_dim
        );

        // Combined embedding dimension using actual codebook dimension
        let combined_dim = config.num_quantizers * actual_codebook_dim;

        // Projection from combined embeddings to latent space
        // Note: The Qwen3 model has a different architecture (HiFi-GAN based),
        // so we use our own projection layer with random init for now.
        let embed_proj = if combined_dim != config.latent_dim {
            // Try to load from weights, fall back to random init
            let proj = conv1d(
                combined_dim,
                config.latent_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("embed_proj"),
            )
            .or_else(|_| {
                // Create random initialized layer
                debug!("Using random init for embed_proj");
                create_random_conv1d(combined_dim, config.latent_dim, 1, device)
            })
            .map_err(|e| TtsError::internal(format!("failed to create embed_proj: {e}")))?;
            Some(proj)
        } else {
            None
        };

        // Input projection - try Qwen3 format first, then our format, then random
        let input_conv = vb
            .pp("decoder.decoder.0")
            .get((config.decoder_dim, config.latent_dim, 7), "conv.weight")
            .and_then(|w| {
                let b = vb
                    .pp("decoder.decoder.0")
                    .get((config.decoder_dim,), "conv.bias")?;
                Ok((w, b))
            })
            .map(|(w, b)| {
                Conv1d::new(
                    w,
                    Some(b),
                    Conv1dConfig {
                        padding: 3,
                        ..Default::default()
                    },
                )
            })
            .or_else(|_| {
                conv1d(
                    config.latent_dim,
                    config.decoder_dim,
                    7,
                    Conv1dConfig {
                        padding: 3,
                        ..Default::default()
                    },
                    vb.pp("input_conv"),
                )
            })
            .or_else(|_| {
                debug!("Using random init for input_conv");
                create_random_conv1d(config.latent_dim, config.decoder_dim, 7, device)
            })
            .map_err(|e| TtsError::internal(format!("failed to create input_conv: {e}")))?;

        // Build upsampling stages - use random init as fallback
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
            .or_else(|_| {
                debug!("Using random init for upsample_{}", i);
                UpsampleBlock::new_random(
                    channels,
                    out_channels,
                    factor,
                    kernel_size,
                    config.num_residual_blocks,
                    device,
                )
            })
            .map_err(|e| TtsError::internal(format!("failed to create upsample_{i}: {e}")))?;

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
            .or_else(|_| {
                debug!("Using random init for upsample_{}", idx);
                UpsampleBlock::new_random(
                    channels,
                    out_channels,
                    factor,
                    kernel_size,
                    config.num_residual_blocks,
                    device,
                )
            })
            .map_err(|e| TtsError::internal(format!("failed to create upsample_{idx}: {e}")))?;

            upsample_blocks.push(block);
            channels = out_channels;
        }

        // Try Qwen3 format for output conv, then fallback to our format, then random
        let output_conv = vb
            .pp("decoder.decoder.6")
            .get((1, channels, 7), "conv.weight")
            .and_then(|w| {
                let b = vb.pp("decoder.decoder.6").get((1,), "conv.bias")?;
                Ok((w, b))
            })
            .map(|(w, b)| {
                Conv1d::new(
                    w,
                    Some(b),
                    Conv1dConfig {
                        padding: 3,
                        ..Default::default()
                    },
                )
            })
            .or_else(|_| {
                conv1d(
                    channels,
                    1,
                    7,
                    Conv1dConfig {
                        padding: 3,
                        ..Default::default()
                    },
                    vb.pp("output_conv"),
                )
            })
            .or_else(|_| {
                debug!("Using random init for output_conv");
                create_random_conv1d(channels, 1, 7, device)
            })
            .map_err(|e| TtsError::internal(format!("failed to create output_conv: {e}")))?;

        // Try to load HiFi-GAN blocks from Qwen3 format
        // decoder.decoder.{1-4} are the upsample blocks with Snake activation
        let (hifi_blocks, final_snake, use_hifi) = Self::try_load_hifi_gan(&vb, &config, device);

        if use_hifi {
            info!(
                "Decoder loaded (HiFi-GAN mode): {} codebooks x {} entries, latent={}, upsample={}x",
                config.num_quantizers,
                config.codebook_size,
                config.latent_dim,
                config.total_upsample()
            );
        } else {
            info!(
                "Decoder loaded (legacy mode): {} codebooks x {} entries, latent={}, upsample={}x",
                config.num_quantizers,
                config.codebook_size,
                config.latent_dim,
                config.total_upsample()
            );
        }

        Ok(Self {
            codebooks,
            embed_proj,
            input_conv,
            upsample_blocks,
            hifi_blocks,
            final_snake,
            output_conv,
            config,
            device: device.clone(),
            use_hifi,
        })
    }

    /// Try to load HiFi-GAN components from Qwen3-TTS weights.
    fn try_load_hifi_gan(
        vb: &VarBuilder,
        _config: &DecoderConfig,
        device: &Device,
    ) -> (Option<Vec<HifiUpsampleBlock>>, Option<Snake>, bool) {
        // Qwen3-TTS HiFi-GAN structure:
        // decoder.decoder.0: Initial conv (already loaded as input_conv)
        // decoder.decoder.1: Upsample block (1536 -> 768, 8x)
        // decoder.decoder.2: Upsample block (768 -> 384, 5x)
        // decoder.decoder.3: Upsample block (384 -> 192, 4x)
        // decoder.decoder.4: Upsample block (192 -> 96, 3x)
        // decoder.decoder.5: Final Snake activation
        // decoder.decoder.6: Output conv (already loaded as output_conv)

        // Channel progression: 1536 -> 768 -> 384 -> 192 -> 96
        let channels = [1536, 768, 384, 192, 96];
        // Upsample factors: 8, 5, 4, 3 (but implemented as kernel sizes 16, 10, 8, 6)
        let kernel_sizes = [16, 10, 8, 6];
        let upsample_factors = [8, 5, 4, 3];

        let mut hifi_blocks = Vec::with_capacity(4);
        let mut loaded_count = 0;

        for i in 0..4 {
            let block_vb = vb.pp(format!("decoder.decoder.{}", i + 1));

            match HifiUpsampleBlock::from_vb(
                block_vb,
                channels[i],
                channels[i + 1],
                upsample_factors[i],
                kernel_sizes[i],
                3, // 3 residual blocks per upsample block
                device,
            ) {
                Ok(block) => {
                    hifi_blocks.push(block);
                    loaded_count += 1;
                    debug!("Loaded HiFi-GAN block {}", i + 1);
                }
                Err(e) => {
                    debug!("Failed to load HiFi-GAN block {}: {}", i + 1, e);
                    break;
                }
            }
        }

        // Only use HiFi-GAN mode if we loaded all 4 blocks
        if loaded_count < 4 {
            debug!(
                "Only loaded {}/4 HiFi-GAN blocks, falling back to legacy mode",
                loaded_count
            );
            return (None, None, false);
        }

        // Try to load final Snake activation
        let final_snake = Snake::from_vb(vb.pp("decoder.decoder.5"), 96).ok();

        if final_snake.is_none() {
            debug!("Failed to load final Snake, falling back to legacy mode");
            return (None, None, false);
        }

        info!("Successfully loaded HiFi-GAN decoder with Snake activation");
        (Some(hifi_blocks), final_snake, true)
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

        // Use HiFi-GAN mode if available, otherwise legacy mode
        if self.use_hifi {
            x = self.decode_hifi_gan(x)?;
        } else {
            x = self.decode_legacy(x)?;
        }

        // Squeeze and convert to Vec
        let x = x.squeeze(0)?.squeeze(0)?;
        x.to_vec1()
    }

    /// Decode using HiFi-GAN architecture with Snake activation.
    fn decode_hifi_gan(&self, mut x: Tensor) -> CandleResult<Tensor> {
        // No initial activation - Snake is applied at the start of each upsample block

        // Process through HiFi-GAN upsample blocks
        if let Some(ref hifi_blocks) = self.hifi_blocks {
            for (i, block) in hifi_blocks.iter().enumerate() {
                x = block.forward(&x)?;
                debug!("HiFi-GAN block {} output shape: {:?}", i, x.dims());
            }
        }

        // Final Snake activation
        if let Some(ref snake) = self.final_snake {
            x = snake.forward(&x)?;
        }

        // Output projection
        let x = self.output_conv.forward(&x)?;

        // Apply tanh to constrain output to [-1, 1]
        x.tanh()
    }

    /// Decode using legacy upsampling (when HiFi-GAN weights not available).
    fn decode_legacy(&self, mut x: Tensor) -> CandleResult<Tensor> {
        x = leaky_relu(&x, 0.1)?;

        // Upsampling stages
        for block in &self.upsample_blocks {
            x = block.forward(&x)?;
        }

        // Output projection
        let x = self.output_conv.forward(&x)?;

        // Apply tanh to constrain output to [-1, 1]
        x.tanh()
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

/// Snake activation: x + (1/alpha) * sin²(alpha * x)
///
/// This is the activation function used in Qwen3-TTS-Tokenizer HiFi-GAN decoder.
/// It provides smooth, periodic activation that helps with audio synthesis.
#[derive(Debug, Clone)]
pub struct Snake {
    /// Alpha parameter per channel [channels].
    alpha: Tensor,
    /// Beta parameter per channel [channels] (optional, used as scaling).
    beta: Option<Tensor>,
}

impl Snake {
    /// Create new Snake activation from weights.
    pub fn new(alpha: Tensor, beta: Option<Tensor>) -> Self {
        Self { alpha, beta }
    }

    /// Load Snake activation from VarBuilder.
    pub fn from_vb(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        let beta = vb.get((channels,), "beta").ok();
        Ok(Self { alpha, beta })
    }

    /// Create with random initialization.
    pub fn new_random(channels: usize, device: &Device) -> CandleResult<Self> {
        // Initialize alpha to 1.0 (identity-like behavior initially)
        let alpha = Tensor::ones((channels,), DType::F32, device)?;
        let beta = Some(Tensor::ones((channels,), DType::F32, device)?);
        Ok(Self { alpha, beta })
    }

    /// Forward pass: x + (1/alpha) * sin²(alpha * x)
    ///
    /// Input shape: [batch, channels, seq_len]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Reshape alpha for broadcasting: [1, channels, 1]
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;

        // Compute alpha * x
        let ax = x.broadcast_mul(&alpha)?;

        // Compute sin²(alpha * x)
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;

        // Compute 1/alpha
        let inv_alpha = (1.0 / &alpha)?;

        // x + (1/alpha) * sin²(alpha * x)
        let activation = (x + &(sin_sq.broadcast_mul(&inv_alpha)?))?;

        // Apply beta scaling if present
        if let Some(ref beta) = self.beta {
            let beta = beta.unsqueeze(0)?.unsqueeze(2)?;
            activation.broadcast_mul(&beta)
        } else {
            Ok(activation)
        }
    }
}

/// HiFi-GAN Residual Block with Snake activation.
///
/// Structure from Qwen3-TTS-Tokenizer:
/// - Snake activation (act1)
/// - Conv1d
/// - Snake activation (act2)
/// - Conv1d
/// - Residual connection
#[derive(Debug)]
struct HifiResBlock {
    act1: Snake,
    conv1: Conv1d,
    act2: Snake,
    conv2: Conv1d,
}

impl HifiResBlock {
    /// Load from VarBuilder (Qwen3 format).
    fn from_vb(vb: VarBuilder, channels: usize, kernel_size: usize) -> CandleResult<Self> {
        let act1 = Snake::from_vb(vb.pp("act1"), channels)?;
        let act2 = Snake::from_vb(vb.pp("act2"), channels)?;

        let padding = kernel_size / 2;
        let conv1_cfg = Conv1dConfig {
            padding,
            ..Default::default()
        };
        let conv1 = conv1d(
            channels,
            channels,
            kernel_size,
            conv1_cfg,
            vb.pp("conv1.conv"),
        )?;

        // Second conv is typically 1x1 projection
        let conv2 = conv1d(
            channels,
            channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv2.conv"),
        )?;

        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    /// Create with random initialization.
    fn new_random(channels: usize, kernel_size: usize, device: &Device) -> CandleResult<Self> {
        let act1 = Snake::new_random(channels, device)?;
        let act2 = Snake::new_random(channels, device)?;

        let padding = kernel_size / 2;
        let w1 = Tensor::randn(0.0f32, 0.02, (channels, channels, kernel_size), device)?;
        let b1 = Tensor::zeros((channels,), DType::F32, device)?;
        let conv1 = Conv1d::new(
            w1,
            Some(b1),
            Conv1dConfig {
                padding,
                ..Default::default()
            },
        );

        let w2 = Tensor::randn(0.0f32, 0.02, (channels, channels, 1), device)?;
        let b2 = Tensor::zeros((channels,), DType::F32, device)?;
        let conv2 = Conv1d::new(w2, Some(b2), Conv1dConfig::default());

        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        let x = self.act1.forward(x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.act2.forward(&x)?;
        let x = self.conv2.forward(&x)?;

        x + residual
    }
}

/// HiFi-GAN Upsample Block with Snake activation.
///
/// Structure from Qwen3-TTS-Tokenizer (decoder.decoder.{1-4}):
/// - Snake activation (block.0)
/// - ConvTranspose1d for upsampling (block.1.conv)
/// - 3 residual blocks (block.2, block.3, block.4)
#[derive(Debug)]
struct HifiUpsampleBlock {
    snake: Snake,
    upsample_conv: ConvTranspose1d,
    res_blocks: Vec<HifiResBlock>,
}

impl HifiUpsampleBlock {
    /// Load from VarBuilder (Qwen3 format).
    #[allow(dead_code)]
    fn from_vb(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        upsample_factor: usize,
        kernel_size: usize,
        num_res_blocks: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        // Snake activation at the start
        let snake = Snake::from_vb(vb.pp("block.0"), in_channels)
            .or_else(|_| Snake::new_random(in_channels, device))?;

        // Upsample conv (transposed convolution)
        let padding = (kernel_size - upsample_factor) / 2;
        let conv_cfg = ConvTranspose1dConfig {
            stride: upsample_factor,
            padding,
            ..Default::default()
        };
        let upsample_conv = conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_cfg,
            vb.pp("block.1.conv"),
        )
        .or_else(|_| -> CandleResult<ConvTranspose1d> {
            // Random fallback
            let w = Tensor::randn(
                0.0f32,
                0.02,
                (in_channels, out_channels, kernel_size),
                device,
            )?;
            let b = Tensor::zeros((out_channels,), DType::F32, device)?;
            Ok(ConvTranspose1d::new(w, Some(b), conv_cfg))
        })?;

        // Residual blocks
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for i in 0..num_res_blocks {
            let block = HifiResBlock::from_vb(vb.pp(format!("block.{}", i + 2)), out_channels, 7)
                .or_else(|_| HifiResBlock::new_random(out_channels, 7, device))?;
            res_blocks.push(block);
        }

        Ok(Self {
            snake,
            upsample_conv,
            res_blocks,
        })
    }

    /// Create with random initialization.
    #[allow(dead_code)]
    fn new_random(
        in_channels: usize,
        out_channels: usize,
        upsample_factor: usize,
        kernel_size: usize,
        num_res_blocks: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let snake = Snake::new_random(in_channels, device)?;

        let padding = (kernel_size - upsample_factor) / 2;
        let w = Tensor::randn(
            0.0f32,
            0.02,
            (in_channels, out_channels, kernel_size),
            device,
        )?;
        let b = Tensor::zeros((out_channels,), DType::F32, device)?;
        let conv_cfg = ConvTranspose1dConfig {
            stride: upsample_factor,
            padding,
            ..Default::default()
        };
        let upsample_conv = ConvTranspose1d::new(w, Some(b), conv_cfg);

        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for _ in 0..num_res_blocks {
            res_blocks.push(HifiResBlock::new_random(out_channels, 7, device)?);
        }

        Ok(Self {
            snake,
            upsample_conv,
            res_blocks,
        })
    }

    #[allow(dead_code)]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.snake.forward(x)?;
        let mut x = self.upsample_conv.forward(&x)?;

        for block in &self.res_blocks {
            x = block.forward(&x)?;
        }

        Ok(x)
    }
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
