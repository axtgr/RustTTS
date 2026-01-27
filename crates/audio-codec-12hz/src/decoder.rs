//! Neural codec decoder for Qwen3-TTS-Tokenizer-12Hz.
//!
//! This module implements the audio decoder that converts acoustic tokens
//! back to PCM waveform using a neural network architecture.
//!
//! The Qwen3-TTS-Tokenizer-12Hz decoder architecture:
//! 1. **RVQ Dequantize**: tokens → embeddings via codebook lookup
//! 2. **Output projection**: each codebook embedding projected to 512-dim
//! 3. **Residual sum**: sum of all 16 projected embeddings
//! 4. **Pre-conv**: Conv1d [1024, 512, 3] projection
//! 5. **Pre-Transformer**: 8-layer causal transformer with layer scale
//! 6. **HiFi-GAN decoder**: 4 upsample blocks with Snake activation
//! 7. **Output**: Final conv to mono audio

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder,
    conv_transpose1d, conv1d,
};
use serde_json::Value;
use tracing::{debug, info, instrument};

use tts_core::{TtsError, TtsResult};

/// Debug print macro - only active when `debug_decoder` feature is enabled.
#[cfg(feature = "debug_decoder")]
macro_rules! debug_print {
    ($($arg:tt)*) => { debug_print!($($arg)*) };
}

#[cfg(not(feature = "debug_decoder"))]
macro_rules! debug_print {
    ($($arg:tt)*) => {};
}

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
/// The decoder uses RVQ dequantization with residual sum, then
/// processes through Pre-Transformer and HiFi-GAN to produce audio.
///
/// Architecture:
/// 1. RVQ dequantize: lookup embeddings from 16 codebooks
/// 2. Output projection: project each embedding 256→512 and sum
/// 3. Pre-conv: Conv1d [1024, 512, 3]
/// 4. Pre-Transformer: 8-layer causal transformer with final norm
/// 5. ConvNeXt Upsample: 2 blocks, each 2x upsample (total 4x)
/// 6. HiFi-GAN: 4 upsample blocks with Snake activation (total 480x)
/// 7. Output: Conv1d to mono audio
///
/// Total upsampling: 4 (ConvNeXt) × 480 (HiFi-GAN) = 1920x
#[derive(Debug)]
pub struct NeuralDecoder {
    /// Codebook embeddings (one per quantizer) [2048, 256].
    codebooks: Vec<Tensor>,
    /// RVQ output projection for first quantizer (256→512).
    rvq_first_output_proj: Option<RvqOutputProj>,
    /// RVQ output projection for rest quantizers (256→512).
    rvq_rest_output_proj: Option<RvqOutputProj>,
    /// Pre-conv: [1024, 512, 3] projects sum to 1024-dim (causal).
    pre_conv: Option<CausalConv1d>,
    /// Pre-Transformer: 8-layer transformer with final norm.
    pre_transformer: Option<PreTransformer>,
    /// ConvNeXt upsample blocks (2 blocks, each 2x = 4x total).
    convnext_upsample: Option<Vec<ConvNeXtUpsampleBlock>>,
    /// HiFi-GAN initial conv [1536, 1024, 7] (causal).
    input_conv: CausalConv1d,
    /// Upsampling stages (legacy mode).
    upsample_blocks: Vec<UpsampleBlock>,
    /// HiFi-GAN upsample blocks (real Qwen3 mode).
    hifi_blocks: Option<Vec<HifiUpsampleBlock>>,
    /// Final Snake activation (for HiFi-GAN mode).
    final_snake: Option<Snake>,
    /// Final output convolution (causal).
    output_conv: CausalConv1d,
    /// Configuration.
    config: DecoderConfig,
    /// Device.
    device: Device,
    /// Whether using full Qwen3 mode (with pre_transformer).
    use_full_qwen3: bool,
    /// Whether using HiFi-GAN mode.
    use_hifi: bool,
    /// Legacy projection from combined codebook embeddings.
    embed_proj: Option<Conv1d>,
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

        // Input projection from latent to decoder dimension (causal)
        let input_conv = conv1d(
            config.latent_dim,
            config.decoder_dim,
            7,
            Conv1dConfig::default(),
            vb.pp("input_conv"),
        )
        .map(|conv| {
            let weight = conv.weight().clone();
            let bias = conv.bias().cloned();
            CausalConv1d::new(weight, bias, 7)
        })?;

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
            Conv1dConfig::default(),
            vb.pp("output_conv"),
        )
        .map(|conv| {
            let weight = conv.weight().clone();
            let bias = conv.bias().cloned();
            CausalConv1d::new(weight, bias, 7)
        })?;

        Ok(Self {
            codebooks,
            rvq_first_output_proj: None,
            rvq_rest_output_proj: None,
            pre_conv: None,
            pre_transformer: None,
            convnext_upsample: None,
            embed_proj: Some(embed_proj.unwrap_or_else(|| {
                // Fallback - create identity-like projection
                let w = Tensor::zeros(
                    (config.latent_dim, config.latent_dim, 1),
                    DType::F32,
                    device,
                )
                .unwrap();
                let b = Tensor::zeros((config.latent_dim,), DType::F32, device).unwrap();
                Conv1d::new(w, Some(b), Conv1dConfig::default())
            })),
            input_conv,
            upsample_blocks,
            hifi_blocks: None,
            final_snake: None,
            output_conv,
            config,
            device: device.clone(),
            use_full_qwen3: false,
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
    /// Loads 16 codebook embeddings and the full Qwen3-TTS decoder architecture.
    /// Falls back to legacy mode if pre_transformer weights aren't available.
    pub fn from_vb(vb: VarBuilder, config: DecoderConfig, device: &Device) -> TtsResult<Self> {
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
            codebooks[0].dim(1).unwrap_or(256)
        } else {
            256 // Qwen3-TTS uses 256-dim codebook embeddings
        };

        info!(
            "Loaded {} codebooks ({} x {})",
            codebooks.len(),
            config.codebook_size,
            actual_codebook_dim
        );

        // Try to load RVQ output projections (256 → 512)
        let rvq_first_output_proj =
            RvqOutputProj::from_vb(vb.pp("decoder.quantizer.rvq_first.output_proj")).ok();

        let rvq_rest_output_proj =
            RvqOutputProj::from_vb(vb.pp("decoder.quantizer.rvq_rest.output_proj")).ok();

        let has_rvq_proj = rvq_first_output_proj.is_some() && rvq_rest_output_proj.is_some();
        if has_rvq_proj {
            debug!("Loaded RVQ output projections (256 → 512)");
        }

        // Try to load pre_conv: [1024, 512, 3]
        // This is a CausalConvNet which uses LEFT-only padding
        let pre_conv = vb
            .pp("decoder.pre_conv.conv")
            .get((1024, 512, 3), "weight")
            .and_then(|w| {
                let b = vb.pp("decoder.pre_conv.conv").get((1024,), "bias")?;
                Ok(CausalConv1d::new(w, Some(b), 3))
            })
            .ok();

        if pre_conv.is_some() {
            debug!("Loaded pre_conv [1024, 512, 3] (causal)");
        }

        // Try to load pre_transformer (8-layer transformer)
        let pre_transformer =
            PreTransformer::from_vb(vb.pp("decoder.pre_transformer"), &config, device).ok();

        // Try to load ConvNeXt upsample blocks (decoder.upsample.{0,1})
        let convnext_upsample = Self::try_load_convnext_upsample(&vb, device);
        let has_convnext = convnext_upsample.is_some();
        if has_convnext {
            debug!("Loaded ConvNeXt upsample blocks (4x upsampling)");
        }

        let use_full_qwen3 =
            pre_conv.is_some() && pre_transformer.is_some() && has_rvq_proj && has_convnext;
        if use_full_qwen3 {
            info!("Using full Qwen3-TTS decoder architecture with ConvNeXt upsample");
        } else {
            debug!("Pre-transformer not fully loaded, will use legacy mode");
        }

        // Legacy embed_proj for fallback mode
        let embed_proj = if !use_full_qwen3 {
            let combined_dim = config.num_quantizers * actual_codebook_dim;
            if combined_dim != config.latent_dim {
                debug!("Using legacy embed_proj (random init)");
                Some(
                    create_random_conv1d(combined_dim, config.latent_dim, 1, device).map_err(
                        |e| TtsError::internal(format!("failed to create embed_proj: {e}")),
                    )?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // HiFi-GAN input conv: decoder.decoder.0 [1536, 1024, 7] (CausalConvNet)
        let input_conv = vb
            .pp("decoder.decoder.0")
            .get((config.decoder_dim, config.latent_dim, 7), "conv.weight")
            .and_then(|w| {
                let b = vb
                    .pp("decoder.decoder.0")
                    .get((config.decoder_dim,), "conv.bias")?;
                Ok(CausalConv1d::new(w, Some(b), 7))
            })
            .or_else(|_| {
                debug!("Using random init for input_conv");
                create_random_conv1d(config.latent_dim, config.decoder_dim, 7, device).map(|conv| {
                    // Convert Conv1d to CausalConv1d
                    let weight = conv.weight().clone();
                    let bias = conv.bias().cloned();
                    CausalConv1d::new(weight, bias, 7)
                })
            })
            .map_err(|e| TtsError::internal(format!("failed to create input_conv: {e}")))?;

        // Try to load HiFi-GAN blocks from Qwen3 format FIRST
        // decoder.decoder.{1-4} are the upsample blocks with Snake activation
        let (hifi_blocks, final_snake, use_hifi) = Self::try_load_hifi_gan(&vb, &config, device);

        // Determine output channels based on whether HiFi-GAN was loaded
        // HiFi-GAN outputs 96 channels, legacy outputs variable channels
        let (upsample_blocks, output_channels) = if use_hifi {
            // HiFi-GAN mode - output is 96 channels, don't need legacy blocks
            (Vec::new(), 96usize)
        } else {
            // Legacy mode - build upsampling stages
            let mut blocks = Vec::new();
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

                blocks.push(block);
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

                blocks.push(block);
                channels = out_channels;
            }
            (blocks, channels)
        };

        // Load output conv with correct number of input channels (CausalConvNet)
        let output_conv = vb
            .pp("decoder.decoder.6")
            .get((1, output_channels, 7), "conv.weight")
            .and_then(|w| {
                let b = vb.pp("decoder.decoder.6").get((1,), "conv.bias")?;
                Ok(CausalConv1d::new(w, Some(b), 7))
            })
            .or_else(|_| {
                conv1d(
                    output_channels,
                    1,
                    7,
                    Conv1dConfig::default(),
                    vb.pp("output_conv"),
                )
                .map(|conv| {
                    let weight = conv.weight().clone();
                    let bias = conv.bias().cloned();
                    CausalConv1d::new(weight, bias, 7)
                })
            })
            .or_else(|_| {
                debug!("Using random init for output_conv");
                create_random_conv1d(output_channels, 1, 7, device).map(|conv| {
                    let weight = conv.weight().clone();
                    let bias = conv.bias().cloned();
                    CausalConv1d::new(weight, bias, 7)
                })
            })
            .map_err(|e| TtsError::internal(format!("failed to create output_conv: {e}")))?;

        if use_full_qwen3 && use_hifi {
            info!(
                "Decoder loaded (full Qwen3 mode): {} codebooks x {} entries, latent={}, upsample={}x, with pre_transformer",
                config.num_quantizers,
                config.codebook_size,
                config.latent_dim,
                config.total_upsample()
            );
        } else if use_hifi {
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
            rvq_first_output_proj,
            rvq_rest_output_proj,
            pre_conv,
            pre_transformer,
            convnext_upsample,
            embed_proj,
            input_conv,
            upsample_blocks,
            hifi_blocks,
            final_snake,
            output_conv,
            config,
            device: device.clone(),
            use_full_qwen3,
            use_hifi,
        })
    }

    /// Try to load ConvNeXt upsample blocks from Qwen3-TTS weights.
    ///
    /// These blocks provide 4x upsampling (2 blocks, each 2x) between
    /// pre_transformer and HiFi-GAN decoder.
    fn try_load_convnext_upsample(
        vb: &VarBuilder,
        _device: &Device,
    ) -> Option<Vec<ConvNeXtUpsampleBlock>> {
        // decoder.upsample.{0,1} - two ConvNeXt upsample blocks
        // Each block: ConvTranspose1d (2x) + ConvNeXtBlock
        // Total: 2x * 2x = 4x upsampling

        let mut blocks = Vec::with_capacity(2);

        for i in 0..2 {
            match ConvNeXtUpsampleBlock::from_vb(vb.pp(format!("decoder.upsample.{}", i)), 1024) {
                Ok(block) => {
                    blocks.push(block);
                    debug!("Loaded ConvNeXt upsample block {}", i);
                }
                Err(e) => {
                    debug!("Failed to load ConvNeXt upsample block {}: {}", i, e);
                    return None;
                }
            }
        }

        info!(
            "Loaded {} ConvNeXt upsample blocks (4x total upsampling)",
            blocks.len()
        );
        Some(blocks)
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
        // Use full Qwen3 mode if available
        if self.use_full_qwen3 {
            return self.decode_qwen3_internal(tokens);
        }

        // Legacy mode: concatenate embeddings and project
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

    /// Full Qwen3-TTS decoder pipeline.
    ///
    /// Architecture:
    /// 1. RVQ dequantize: lookup embeddings from each codebook
    /// 2. Output projection: project each embedding 256→512 via Conv1d
    /// 3. Residual sum: sum all 16 projected embeddings
    /// 4. Pre-conv: Conv1d [1024, 512, 3]
    /// 5. Pre-Transformer: 8-layer causal transformer with final norm
    /// 6. ConvNeXt Upsample: 2 blocks, each 2x (total 4x upsampling)
    /// 7. HiFi-GAN: 4 upsample blocks with Snake activation → audio
    fn decode_qwen3_internal(&self, tokens: &[Vec<u32>]) -> CandleResult<Vec<f32>> {
        let seq_len = tokens[0].len();
        debug!(
            "Qwen3 decode: {} codebooks x {} tokens",
            tokens.len(),
            seq_len
        );

        // Get required components
        let rvq_first_proj = self.rvq_first_output_proj.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("rvq_first_output_proj not loaded".to_string())
        })?;
        let rvq_rest_proj = self.rvq_rest_output_proj.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("rvq_rest_output_proj not loaded".to_string())
        })?;
        let pre_conv = self
            .pre_conv
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("pre_conv not loaded".to_string()))?;
        let pre_transformer = self
            .pre_transformer
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("pre_transformer not loaded".to_string()))?;
        let convnext_upsample = self
            .convnext_upsample
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("convnext_upsample not loaded".to_string()))?;

        // Step 1 & 2: RVQ dequantize with proper output projection order
        //
        // Python architecture (SplitResidualVectorQuantizer):
        // - rvq_first processes codebook 0 only
        // - rvq_rest processes codebooks 1-15
        //
        // Inside ResidualVectorQuantizer.decode():
        // 1. For each codebook: lookup embedding, transpose(1,2) → [batch, 256, seq]
        // 2. Sum all codebook embeddings WITHIN the same RVQ
        // 3. Apply output_proj (Conv1d) AFTER the sum
        //
        // So the correct order is:
        // rvq_first: cb0 → transpose → output_proj
        // rvq_rest: sum(cb1..cb15) → transpose → output_proj
        // final: rvq_first + rvq_rest

        // Process codebook 0 (rvq_first)
        let token_ids_0: Vec<i64> = tokens[0].iter().map(|&t| t as i64).collect();
        let token_tensor_0 = Tensor::new(token_ids_0.as_slice(), &self.device)?;
        let emb_0 = self.codebooks[0].index_select(&token_tensor_0, 0)?; // [seq, 256]

        // Debug: print first embedding values for verification
        {
            let emb_first_frame: Vec<f32> = emb_0.i(0)?.to_vec1()?;
            debug_print!(
                "DEBUG: cb0 embedding[0, :5] (token {}): {:?}",
                tokens[0][0],
                &emb_first_frame[..5.min(emb_first_frame.len())]
            );
            // Expected: [27.099, -18.818, -6.412, 4.171, 6.394] for token 1995
        }

        let emb_0 = emb_0.unsqueeze(0)?; // [1, seq, 256]
        // Transpose to [1, 256, seq] for Conv1d
        let emb_0_t = emb_0.transpose(1, 2)?;
        // Apply output_proj: [1, 256, seq] → [1, 512, seq]
        let rvq_first_out = rvq_first_proj.forward_conv(&emb_0_t)?;

        // Debug: print projected values
        {
            let proj_first: Vec<f32> = rvq_first_out.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: rvq_first projected[:5, 0]: {:?}",
                &proj_first[..5.min(proj_first.len())]
            );
            // Expected: [-6.599, -5.657, 45.555, -5.767, -2.078]
        }

        debug!("rvq_first output: {:?}", rvq_first_out.dims());

        // Process codebooks 1-15 (rvq_rest)
        // First sum all embeddings, then apply output_proj
        let mut rvq_rest_sum: Option<Tensor> = None;
        for (i, (codebook, codebook_tokens)) in
            self.codebooks.iter().zip(tokens.iter()).enumerate().skip(1)
        {
            let token_ids: Vec<i64> = codebook_tokens.iter().map(|&t| t as i64).collect();
            let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?;
            // Lookup: [seq_len, 256]
            let emb = codebook.index_select(&token_tensor, 0)?;
            // Add batch dimension: [1, seq_len, 256]
            let emb = emb.unsqueeze(0)?;
            // Transpose to [1, 256, seq]
            let emb_t = emb.transpose(1, 2)?;

            // Sum embeddings (residual sum within rvq_rest)
            rvq_rest_sum = match rvq_rest_sum {
                Some(s) => Some(s.add(&emb_t)?),
                None => Some(emb_t),
            };

            if i == 1 {
                // Can't use emb_t here since it's moved, but that's fine
                debug!("rvq_rest codebook {} processed", i);
            }
        }

        // Apply output_proj to the sum
        let rvq_rest_out = if let Some(ref sum) = rvq_rest_sum {
            debug!("rvq_rest sum before proj: {:?}", sum.dims());
            let proj = rvq_rest_proj.forward_conv(sum)?;
            // Debug: print projected values
            {
                let proj_first: Vec<f32> = proj.i((0, .., 0))?.to_vec1()?;
                debug_print!(
                    "DEBUG: rvq_rest projected[:5, 0]: {:?}",
                    &proj_first[..5.min(proj_first.len())]
                );
                // Expected: [14.281, 2.997, -2.655, -1.107, 9.266]
            }
            proj
        } else {
            // If no rest codebooks, create zeros
            Tensor::zeros((1, 512, seq_len), candle_core::DType::F32, &self.device)?
        };
        debug!("rvq_rest output: {:?}", rvq_rest_out.dims());

        // Final sum: rvq_first + rvq_rest
        let x = (rvq_first_out + rvq_rest_out)?;
        // Debug: print final quantizer output
        {
            let final_first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: quantizer output[:5, 0]: {:?}",
                &final_first[..5.min(final_first.len())]
            );
            // Expected: [7.681, -2.659, 42.899, -6.874, 7.187]
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug_print!("DEBUG: quantizer output mean={:.4}, std={:.4}", mean, std);
            // Expected: mean=0.1195, std=11.015
        }
        debug!("After RVQ sum: {:?}", x.dims());

        // Debug: check embedding statistics
        {
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug!(
                "RVQ embedding stats - mean: {:.4}, std: {:.4}, min: {:.4}, max: {:.4}",
                mean,
                std,
                x_flat.iter().cloned().fold(f32::INFINITY, f32::min),
                x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // Step 3: Pre-conv [1024, 512, 3]
        let x = pre_conv.forward(&x)?;
        // Debug pre_conv output
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: pre_conv output[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
            // Expected: [0.0212, 0.0001, -0.0005, 0.0001, 0.0503]
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug_print!("DEBUG: pre_conv output mean={:.4}, std={:.4}", mean, std);
            // Expected: mean=0.0034, std=0.1215
        }
        debug!("After pre_conv: {:?}", x.dims());

        // Step 4: Pre-Transformer (includes final norm)
        let x = pre_transformer.forward(&x)?;
        // Debug pre_transformer output
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: pre_transformer output[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug_print!(
                "DEBUG: pre_transformer output mean={:.4}, std={:.4}",
                mean, std
            );
        }
        debug!("After pre_transformer: {:?}", x.dims());

        // Step 5: ConvNeXt upsample (4x total)
        let mut x = x;
        for (i, block) in convnext_upsample.iter().enumerate() {
            x = block.forward(&x)?;
            // Debug: ConvNeXt output stats
            {
                let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
                debug_print!(
                    "DEBUG: convnext_{} output[:5, 0]: {:?}",
                    i,
                    &first[..5.min(first.len())]
                );
                let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
                let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
                let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                    / x_flat.len() as f32)
                    .sqrt();
                debug_print!(
                    "DEBUG: convnext_{} mean={:.6}, std={:.6}, shape={:?}",
                    i,
                    mean,
                    std,
                    x.dims()
                );
            }
            debug!("After ConvNeXt upsample block {}: {:?}", i, x.dims());
        }

        // Step 6: HiFi-GAN decoder
        let mut x = self.input_conv.forward(&x)?;
        // Debug: input_conv output stats
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: input_conv output[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug_print!(
                "DEBUG: input_conv mean={:.6}, std={:.6}, shape={:?}",
                mean,
                std,
                x.dims()
            );
        }
        debug!("After input_conv: {:?}", x.dims());

        x = self.decode_hifi_gan(x)?;

        // Debug: final audio stats before squeeze
        {
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug!(
                "Final audio stats - mean: {:.4}, std: {:.4}, min: {:.4}, max: {:.4}",
                mean,
                std,
                x_flat.iter().cloned().fold(f32::INFINITY, f32::min),
                x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
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
                // Stats before block
                let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
                let std_before: f32 =
                    (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();

                x = block.forward(&x)?;

                // Stats after block
                let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
                let std_after: f32 =
                    (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();
                debug!(
                    "HiFi-GAN block {} output shape: {:?}, std: {:.4} -> {:.4}",
                    i,
                    x.dims(),
                    std_before,
                    std_after
                );
            }
        }

        // Debug: before final snake
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: before_final_snake[:5, 0]: {:?}, shape={:?}",
                &first[..5.min(first.len())],
                x.dims()
            );
        }

        // Final Snake activation
        if let Some(ref snake) = self.final_snake {
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let std_before: f32 =
                (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();

            // Debug: print Snake parameters
            let alpha_flat: Vec<f32> = snake.alpha.to_vec1()?;
            let alpha_exp: Vec<f32> = snake.alpha.exp()?.to_vec1()?;
            debug!(
                "final_snake alpha[:5]: {:?}, exp(alpha)[:5]: {:?}",
                &alpha_flat[..5.min(alpha_flat.len())],
                &alpha_exp[..5.min(alpha_exp.len())]
            );
            if let Some(ref beta) = snake.beta {
                let beta_flat: Vec<f32> = beta.to_vec1()?;
                let beta_exp: Vec<f32> = beta.exp()?.to_vec1()?;
                debug!(
                    "final_snake beta[:5]: {:?}, exp(beta)[:5]: {:?}",
                    &beta_flat[..5.min(beta_flat.len())],
                    &beta_exp[..5.min(beta_exp.len())]
                );
            }

            x = snake.forward(&x)?;

            // Debug: after final snake
            {
                let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
                debug_print!(
                    "DEBUG: after_final_snake[:5, 0]: {:?}",
                    &first[..5.min(first.len())]
                );
            }

            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let std_after: f32 =
                (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();
            debug!(
                "After final Snake: std {:.4} -> {:.4}",
                std_before, std_after
            );
        }

        // Output projection
        {
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let std_before: f32 =
                (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();

            // Debug: check output_conv weights
            let conv_weight = self.output_conv.weight();
            let w_flat: Vec<f32> = conv_weight.flatten_all()?.to_vec1()?;
            let w_std: f32 =
                (w_flat.iter().map(|v| v.powi(2)).sum::<f32>() / w_flat.len() as f32).sqrt();
            debug!(
                "output_conv weight shape: {:?}, std: {:.6}",
                conv_weight.dims(),
                w_std
            );

            // Print first few weight values for comparison with Python
            let w_first7: Vec<f32> = conv_weight.i((0, 0, ..))?.to_vec1()?;
            debug!("output_conv weight[0,0,:7]: {:?}", w_first7);

            // Print input dimensions
            debug!("Input to output_conv shape: {:?}", x.dims());

            // Print first few input values across channels for comparison
            let x_ch0: Vec<f32> = x.i((0, 0, 0..10))?.to_vec1()?;
            let x_ch1: Vec<f32> = x.i((0, 1, 0..10))?.to_vec1()?;
            let x_ch95: Vec<f32> = x.i((0, 95, 0..10))?.to_vec1()?;
            debug!("Input x[0,0,:10]: {:?}", x_ch0);
            debug!("Input x[0,1,:10]: {:?}", x_ch1);
            debug!("Input x[0,95,:10]: {:?}", x_ch95);

            // Check std per channel
            let mut channel_stds = Vec::with_capacity(96);
            for c in 0..96 {
                let ch_data: Vec<f32> = x.i((0, c, ..))?.to_vec1()?;
                let ch_std =
                    (ch_data.iter().map(|v| v.powi(2)).sum::<f32>() / ch_data.len() as f32).sqrt();
                channel_stds.push(ch_std);
            }
            let max_std = channel_stds
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_std = channel_stds.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_std: f32 = channel_stds.iter().sum::<f32>() / 96.0;
            debug!(
                "Channel std distribution: min={:.4}, mean={:.4}, max={:.4}",
                min_std, mean_std, max_std
            );

            // Count channels with std > 10
            let high_std_count = channel_stds.iter().filter(|&&s| s > 10.0).count();
            debug!("Channels with std > 10: {}/96", high_std_count);

            // Also check weight values for channels 1 and 95
            let w_ch1: Vec<f32> = conv_weight.i((0, 1, ..))?.to_vec1()?;
            let w_ch95: Vec<f32> = conv_weight.i((0, 95, ..))?.to_vec1()?;
            debug!("output_conv weight[0,1,:7]: {:?}", w_ch1);
            debug!("output_conv weight[0,95,:7]: {:?}", w_ch95);

            let x_out = self.output_conv.forward(&x)?;

            let x_flat: Vec<f32> = x_out.flatten_all()?.to_vec1()?;
            let std_after: f32 =
                (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();
            debug!(
                "After output_conv: std {:.4} -> {:.4}",
                std_before, std_after
            );

            // Debug: after output_conv, before tanh
            {
                let first: Vec<f32> = x_out.i((0, 0, 0..10))?.to_vec1()?;
                debug_print!("DEBUG: output_conv_out[:10]: {:?}", first);
            }

            // Apply tanh
            let x_tanh = x_out.tanh()?;

            // Debug: after tanh
            {
                let first: Vec<f32> = x_tanh.i((0, 0, 0..10))?.to_vec1()?;
                debug_print!("DEBUG: after_tanh[:10]: {:?}", first);
            }

            let x_flat: Vec<f32> = x_tanh.flatten_all()?.to_vec1()?;
            let std_tanh: f32 =
                (x_flat.iter().map(|v| v.powi(2)).sum::<f32>() / x_flat.len() as f32).sqrt();
            debug!("After tanh: std {:.4}", std_tanh);

            return Ok(x_tanh);
        }
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

    /// Forward pass: x + (1/exp(beta)) * sin²(x * exp(alpha))
    ///
    /// This matches Python SnakeBeta implementation:
    /// alpha = exp(alpha_param)
    /// beta = exp(beta_param)
    /// output = x + (1/beta) * sin²(x * alpha)
    ///
    /// Input shape: [batch, channels, seq_len]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Reshape alpha for broadcasting: [1, channels, 1]
        let alpha_param = self.alpha.unsqueeze(0)?.unsqueeze(2)?;

        // Apply exp to get actual alpha (matching Python: alpha = torch.exp(alpha))
        let alpha = alpha_param.exp()?;

        // Compute x * alpha
        let ax = x.broadcast_mul(&alpha)?;

        // Compute sin²(x * alpha)
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;

        // Get beta and compute 1/beta
        if let Some(ref beta_param) = self.beta {
            // Apply exp to get actual beta (matching Python: beta = torch.exp(beta))
            let beta_param = beta_param.unsqueeze(0)?.unsqueeze(2)?;
            let beta = beta_param.exp()?;

            // Small epsilon to avoid division by zero (matching Python: no_div_by_zero = 1e-9)
            let eps = 1e-9_f64;
            let beta_safe = (beta + eps)?;
            let inv_beta = (1.0 / beta_safe)?;

            // x + (1/beta) * sin²(x * alpha)
            let term = sin_sq.broadcast_mul(&inv_beta)?;
            x + &term
        } else {
            // Fallback: just use 1/alpha (original Snake formula without beta)
            let inv_alpha = (1.0 / &alpha)?;
            let term = sin_sq.broadcast_mul(&inv_alpha)?;
            x + &term
        }
    }
}

// =============================================================================
// Pre-Transformer Components
// =============================================================================

/// RMS Layer Normalization.
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_vb(vb: VarBuilder, dim: usize, eps: f64) -> CandleResult<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, seq_len, hidden]
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // Compute RMS
        let variance = x.sqr()?.mean_keepdim(2)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;

        // Apply weight
        let weight = self.weight.unsqueeze(0)?.unsqueeze(0)?;
        normalized.broadcast_mul(&weight)?.to_dtype(dtype)
    }
}

/// Layer Scale - learned per-channel scaling.
#[derive(Debug, Clone)]
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn from_vb(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        let scale = vb.get((dim,), "scale")?;
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let scale = self.scale.unsqueeze(0)?.unsqueeze(0)?;
        x.broadcast_mul(&scale)
    }
}

/// SwiGLU MLP block.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl MLP {
    fn from_vb(vb: VarBuilder, hidden_size: usize, intermediate_size: usize) -> CandleResult<Self> {
        let gate_proj = vb.get((intermediate_size, hidden_size), "gate_proj.weight")?;
        let up_proj = vb.get((intermediate_size, hidden_size), "up_proj.weight")?;
        let down_proj = vb.get((hidden_size, intermediate_size), "down_proj.weight")?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, seq_len, hidden]
        let (batch, seq_len, hidden_size) = x.dims3()?;
        let _intermediate_size = self.gate_proj.dims()[0];

        // Flatten for matmul
        let x_flat = x.reshape((batch * seq_len, hidden_size))?;

        let gate = x_flat.matmul(&self.gate_proj.t()?)?;
        let up = x_flat.matmul(&self.up_proj.t()?)?;

        // SwiGLU: silu(gate) * up
        let gate_activated = candle_nn::ops::silu(&gate)?;
        let hidden = (gate_activated * up)?;

        let out = hidden.matmul(&self.down_proj.t()?)?;
        out.reshape((batch, seq_len, hidden_size))
    }
}

/// Self-attention block with separate kv_dim support.
/// In Qwen3-TTS Pre-Transformer: hidden_size=512, kv_dim=1024, num_heads=16, head_dim=64
#[derive(Debug)]
struct SelfAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    /// Create from VarBuilder with explicit kv_dim (can differ from hidden_size)
    fn from_vb(
        vb: VarBuilder,
        hidden_size: usize,
        kv_dim: usize,
        num_heads: usize,
    ) -> CandleResult<Self> {
        let head_dim = kv_dim / num_heads;

        let q_proj = vb.get((kv_dim, hidden_size), "q_proj.weight")?;
        let k_proj = vb.get((kv_dim, hidden_size), "k_proj.weight")?;
        let v_proj = vb.get((kv_dim, hidden_size), "v_proj.weight")?;
        let o_proj = vb.get((hidden_size, kv_dim), "o_proj.weight")?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let kv_dim = self.num_heads * self.head_dim;

        // Flatten for matmul: [batch * seq_len, hidden]
        let x_flat = x.reshape((batch * seq_len, hidden))?;

        // Project Q, K, V: [batch * seq_len, hidden] -> [batch * seq_len, kv_dim]
        let q = x_flat.matmul(&self.q_proj.t()?)?;
        let k = x_flat.matmul(&self.k_proj.t()?)?;
        let v = x_flat.matmul(&self.v_proj.t()?)?;

        // Reshape to [batch, seq_len, num_heads, head_dim], then transpose to [batch, num_heads, seq_len, head_dim]
        // Reshape and transpose to [batch, num_heads, seq_len, head_dim]
        // Use contiguous() for Metal GPU compatibility
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?;

        // Causal mask
        let mask = Self::causal_mask(seq_len, x.device())?;
        let scores = scores.broadcast_add(&mask)?;

        // Softmax and apply to values
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, kv_dim]
        let out = out.transpose(1, 2)?.reshape((batch, seq_len, kv_dim))?;

        // Output projection: [batch, seq_len, kv_dim] -> [batch, seq_len, hidden]
        let out_flat = out.reshape((batch * seq_len, kv_dim))?;
        let result = out_flat.matmul(&self.o_proj.t()?)?;
        result.reshape((batch, seq_len, hidden))
    }

    fn causal_mask(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
        let mut mask = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
    }
}

/// Pre-Transformer layer with layer scale.
#[derive(Debug)]
struct PreTransformerLayer {
    input_layernorm: RmsNorm,
    self_attn: SelfAttention,
    self_attn_layer_scale: LayerScale,
    post_attention_layernorm: RmsNorm,
    mlp: MLP,
    mlp_layer_scale: LayerScale,
}

impl PreTransformerLayer {
    fn from_vb(
        vb: VarBuilder,
        hidden_size: usize,
        kv_dim: usize,
        intermediate_size: usize,
        num_heads: usize,
        eps: f64,
    ) -> CandleResult<Self> {
        let input_layernorm = RmsNorm::from_vb(vb.pp("input_layernorm"), hidden_size, eps)?;
        let self_attn = SelfAttention::from_vb(vb.pp("self_attn"), hidden_size, kv_dim, num_heads)?;
        let self_attn_layer_scale =
            LayerScale::from_vb(vb.pp("self_attn_layer_scale"), hidden_size)?;
        let post_attention_layernorm =
            RmsNorm::from_vb(vb.pp("post_attention_layernorm"), hidden_size, eps)?;
        let mlp = MLP::from_vb(vb.pp("mlp"), hidden_size, intermediate_size)?;
        let mlp_layer_scale = LayerScale::from_vb(vb.pp("mlp_layer_scale"), hidden_size)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            self_attn_layer_scale,
            post_attention_layernorm,
            mlp,
            mlp_layer_scale,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Self-attention with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = self.self_attn_layer_scale.forward(&x)?;
        let x = (residual + x)?;

        // MLP with residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = self.mlp_layer_scale.forward(&x)?;
        residual + x
    }
}

/// Pre-Transformer module.
///
/// This is the transformer that processes the quantizer output before HiFi-GAN.
/// Structure: input_proj -> layers -> norm -> output_proj
#[derive(Debug)]
struct PreTransformer {
    input_proj: Tensor,
    input_proj_bias: Tensor,
    layers: Vec<PreTransformerLayer>,
    /// Final RmsNorm before output projection.
    norm: Option<RmsNorm>,
    output_proj: Tensor,
    output_proj_bias: Tensor,
}

impl PreTransformer {
    fn from_vb(vb: VarBuilder, config: &DecoderConfig, _device: &Device) -> CandleResult<Self> {
        // input_proj: [512, 1024] - projects from 1024 (pre_conv output) to 512 (hidden)
        let input_proj = vb.get((512, 1024), "input_proj.weight")?;
        let input_proj_bias = vb.get((512,), "input_proj.bias")?;

        // Load transformer layers
        // Qwen3-TTS Pre-Transformer: hidden_size=512, kv_dim=1024, intermediate=1024, 16 heads
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            match PreTransformerLayer::from_vb(
                vb.pp(format!("layers.{}", i)),
                512,  // hidden_size
                1024, // kv_dim (num_heads * head_dim = 16 * 64 = 1024)
                1024, // intermediate_size
                16,   // num_heads
                config.rms_norm_eps,
            ) {
                Ok(layer) => layers.push(layer),
                Err(e) => {
                    debug!("Failed to load pre_transformer layer {}: {}", i, e);
                    return Err(e);
                }
            }
        }

        // Final norm: [512] - RmsNorm before output projection
        let norm = RmsNorm::from_vb(vb.pp("norm"), 512, config.rms_norm_eps).ok();
        if norm.is_some() {
            debug!("Loaded pre_transformer final norm");
        }

        // output_proj: [1024, 512] - projects from 512 back to 1024
        let output_proj = vb.get((1024, 512), "output_proj.weight")?;
        let output_proj_bias = vb.get((1024,), "output_proj.bias")?;

        info!(
            "Loaded PreTransformer with {} layers{}",
            layers.len(),
            if norm.is_some() {
                " and final norm"
            } else {
                ""
            }
        );

        Ok(Self {
            input_proj,
            input_proj_bias,
            layers,
            norm,
            output_proj,
            output_proj_bias,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, channels=1024, seq_len]
        // Transpose to [batch, seq_len, channels] for transformer
        let x = x.transpose(1, 2)?;
        let (batch, seq_len, _) = x.dims3()?;

        // Input projection: [batch, seq_len, 1024] -> [batch, seq_len, 512]
        // Reshape for matmul compatibility
        let x_flat = x.reshape((batch * seq_len, 1024))?;
        let x_proj = x_flat.matmul(&self.input_proj.t()?)?;
        let x = x_proj.reshape((batch, seq_len, 512))?;
        let bias = self.input_proj_bias.unsqueeze(0)?.unsqueeze(0)?;
        let mut x = x.broadcast_add(&bias)?;

        // Debug: check input_proj output
        {
            let first: Vec<f32> = x.i((0, 0, ..))?.to_vec1()?;
            debug_print!(
                "DEBUG: pre_transformer input_proj[:5]: {:?}",
                &first[..5.min(first.len())]
            );
            // Expected: [-0.00762, -0.11225, -0.01927, 0.01833, 0.01250]
            let x_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let std: f32 = (x_flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / x_flat.len() as f32)
                .sqrt();
            debug_print!("DEBUG: input_proj mean={:.6}, std={:.6}", mean, std);
            // Expected: mean=-0.000834, std=0.055776
        }

        // Transformer layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Final norm before output projection
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
        }

        // Output projection: [batch, seq_len, 512] -> [batch, seq_len, 1024]
        let (batch, seq_len, _) = x.dims3()?;
        let x_flat = x.reshape((batch * seq_len, 512))?;
        let x_proj = x_flat.matmul(&self.output_proj.t()?)?;
        let x = x_proj.reshape((batch, seq_len, 1024))?;
        let bias = self.output_proj_bias.unsqueeze(0)?.unsqueeze(0)?;
        let x = x.broadcast_add(&bias)?;

        // Transpose back to [batch, channels=1024, seq_len]
        x.transpose(1, 2)
    }
}

// =============================================================================
// RVQ Components
// =============================================================================

/// RVQ output projection (Conv1d 1x1).
/// Projects codebook embeddings from 256 to 512 dimensions.
#[derive(Debug)]
struct RvqOutputProj {
    weight: Tensor,
}

impl RvqOutputProj {
    fn from_vb(vb: VarBuilder) -> CandleResult<Self> {
        // Weight shape: [512, 256, 1] (out_channels, in_channels, kernel_size)
        let weight = vb.get((512, 256, 1), "weight")?;
        Ok(Self { weight })
    }

    /// Forward pass for input in [batch, seq_len, 256] format.
    /// Output: [batch, 512, seq_len]
    #[allow(dead_code)]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, seq_len, 256]
        // Conv1d with kernel_size=1 is equivalent to a linear projection
        // weight: [512, 256, 1]
        let weight = self.weight.squeeze(2)?; // [512, 256]

        // For 3D input, we need to use broadcast_matmul or reshape
        // x: [batch, seq_len, 256], weight.t(): [256, 512]
        // Result: [batch, seq_len, 512]
        let (batch, seq_len, _in_dim) = x.dims3()?;
        let x_flat = x.reshape((batch * seq_len, 256))?; // [batch*seq_len, 256]
        let out_flat = x_flat.matmul(&weight.t()?)?; // [batch*seq_len, 512]
        let out = out_flat.reshape((batch, seq_len, 512))?; // [batch, seq_len, 512]
        out.transpose(1, 2) // [batch, 512, seq_len]
    }

    /// Forward pass for input in Conv1d format [batch, 256, seq_len].
    /// Output: [batch, 512, seq_len]
    /// This matches PyTorch Conv1d behavior.
    fn forward_conv(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, 256, seq_len]
        // weight: [512, 256, 1] - Conv1d format (out_channels, in_channels, kernel)
        let weight = self.weight.squeeze(2)?; // [512, 256]

        // Transpose x to [batch, seq_len, 256] for matmul
        let x_t = x.transpose(1, 2)?; // [batch, seq_len, 256]
        let (batch, seq_len, _in_dim) = x_t.dims3()?;

        // Matmul: [batch*seq_len, 256] @ [256, 512] = [batch*seq_len, 512]
        let x_flat = x_t.reshape((batch * seq_len, 256))?;
        let out_flat = x_flat.matmul(&weight.t()?)?;
        let out = out_flat.reshape((batch, seq_len, 512))?; // [batch, seq_len, 512]

        // Transpose back to [batch, 512, seq_len]
        out.transpose(1, 2)
    }
}

// =============================================================================
// CausalConv1d - Conv1d with left-only (causal) padding
// =============================================================================

/// Causal Conv1d - pads on the left side following Qwen3-TTS CausalConvNet.
///
/// For kernel_size=K, dilation=D, stride=S:
/// - effective_kernel = (K - 1) * D + 1
/// - left_padding = effective_kernel - S
/// - May add extra right padding for alignment
#[derive(Debug)]
struct CausalConv1d {
    conv: Conv1d,
    left_padding: usize,
    stride: usize,
    effective_kernel: usize,
}

impl CausalConv1d {
    fn new(weight: Tensor, bias: Option<Tensor>, kernel_size: usize) -> Self {
        Self::new_with_dilation(weight, bias, kernel_size, 1)
    }

    fn new_with_dilation(
        weight: Tensor,
        bias: Option<Tensor>,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        let stride = 1; // Default stride
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let left_padding = effective_kernel - stride;

        let conv = Conv1d::new(
            weight,
            bias,
            Conv1dConfig {
                dilation,
                ..Default::default()
            },
        );

        Self {
            conv,
            left_padding,
            stride,
            effective_kernel,
        }
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Calculate extra padding for proper alignment (matching Python)
        let length = x.dim(2)?;
        let n_frames_float =
            (length + self.left_padding - self.effective_kernel) as f32 / self.stride as f32 + 1.0;
        let n_frames = n_frames_float.ceil() as usize;
        let ideal_length = (n_frames - 1) * self.stride + self.effective_kernel;
        let extra_padding = if ideal_length > length + self.left_padding {
            ideal_length - length - self.left_padding
        } else {
            0
        };

        // Apply padding: left_padding on left, extra_padding on right
        let x = if self.left_padding > 0 || extra_padding > 0 {
            x.pad_with_zeros(2, self.left_padding, extra_padding)?
        } else {
            x.clone()
        };

        self.conv.forward(&x)
    }

    /// Get reference to weight tensor.
    fn weight(&self) -> &Tensor {
        self.conv.weight()
    }
}

// =============================================================================
// ConvNeXt Upsample Components (decoder.upsample.{0,1})
// =============================================================================

/// LayerNorm for ConvNeXt blocks.
///
/// Operates on the channel dimension (last dimension after permute).
#[derive(Debug)]
struct ConvNextLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ConvNextLayerNorm {
    fn from_vb(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        let weight = vb.get((channels,), "weight")?;
        let bias = vb.get((channels,), "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    /// Forward pass for input shape [batch, channels, seq_len].
    ///
    /// Normalizes across the channel dimension.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, channels, seq_len]
        // Transpose to [batch, seq_len, channels] for layer norm
        let x = x.transpose(1, 2)?;

        // Compute mean and variance across channels
        let mean = x.mean_keepdim(2)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(2)?;
        let normalized = x_centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Apply weight and bias
        let weight = self.weight.unsqueeze(0)?.unsqueeze(0)?;
        let bias = self.bias.unsqueeze(0)?.unsqueeze(0)?;
        let out = normalized.broadcast_mul(&weight)?.broadcast_add(&bias)?;

        // Transpose back to [batch, channels, seq_len]
        out.transpose(1, 2)
    }
}

/// ConvNeXt block used in the upsample path.
///
/// Structure (from decoder.upsample.{0,1}.1):
/// - dwconv: Depthwise conv [channels, 1, 7]
/// - norm: LayerNorm [channels]
/// - pwconv1: Pointwise conv (expansion) [4*channels, channels]
/// - GELU activation
/// - pwconv2: Pointwise conv (projection) [channels, 4*channels]
/// - gamma: Residual scale [channels]
/// - Residual connection
#[derive(Debug)]
struct ConvNeXtBlock {
    dwconv_weight: Tensor,
    dwconv_bias: Tensor,
    norm: ConvNextLayerNorm,
    pwconv1_weight: Tensor,
    pwconv1_bias: Tensor,
    pwconv2_weight: Tensor,
    pwconv2_bias: Tensor,
    gamma: Tensor,
    channels: usize,
}

impl ConvNeXtBlock {
    fn from_vb(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        // Depthwise conv: [channels, 1, 7] - groups=channels
        let dwconv_weight = vb.get((channels, 1, 7), "dwconv.conv.weight")?;
        let dwconv_bias = vb.get((channels,), "dwconv.conv.bias")?;

        // LayerNorm
        let norm = ConvNextLayerNorm::from_vb(vb.pp("norm"), channels)?;

        // Pointwise convs (implemented as Linear)
        let intermediate = channels * 4; // 1024 * 4 = 4096
        let pwconv1_weight = vb.get((intermediate, channels), "pwconv1.weight")?;
        let pwconv1_bias = vb.get((intermediate,), "pwconv1.bias")?;
        let pwconv2_weight = vb.get((channels, intermediate), "pwconv2.weight")?;
        let pwconv2_bias = vb.get((channels,), "pwconv2.bias")?;

        // Gamma for residual scaling
        let gamma = vb.get((channels,), "gamma")?;

        Ok(Self {
            dwconv_weight,
            dwconv_bias,
            norm,
            pwconv1_weight,
            pwconv1_bias,
            pwconv2_weight,
            pwconv2_bias,
            gamma,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();
        // x: [batch, channels, seq_len]

        // Depthwise conv with groups=channels
        let x = self.depthwise_conv(x)?;

        // LayerNorm (operates on channel dim)
        let x = self.norm.forward(&x)?;

        // Transpose for pointwise convs: [batch, channels, seq_len] -> [batch, seq_len, channels]
        let x = x.transpose(1, 2)?;
        let (batch, seq_len, _) = x.dims3()?;

        // Pointwise conv1 (expansion)
        let x_flat = x.reshape((batch * seq_len, self.channels))?;
        let x = x_flat.matmul(&self.pwconv1_weight.t()?)?;
        let bias = self.pwconv1_bias.unsqueeze(0)?;
        let x = x.broadcast_add(&bias)?;

        // GELU activation
        let x = x.gelu_erf()?;

        // Pointwise conv2 (projection)
        let x = x.matmul(&self.pwconv2_weight.t()?)?;
        let bias = self.pwconv2_bias.unsqueeze(0)?;
        let x = x.broadcast_add(&bias)?;

        // Reshape back: [batch * seq_len, channels] -> [batch, seq_len, channels]
        let x = x.reshape((batch, seq_len, self.channels))?;

        // Transpose back: [batch, seq_len, channels] -> [batch, channels, seq_len]
        let x = x.transpose(1, 2)?;

        // Apply gamma scaling
        let gamma = self.gamma.unsqueeze(0)?.unsqueeze(2)?; // [1, channels, 1]
        let x = x.broadcast_mul(&gamma)?;

        // Residual connection
        x + residual
    }

    /// Depthwise convolution (groups = channels) with CAUSAL padding.
    fn depthwise_conv(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, channels, seq_len]
        // weight: [channels, 1, 7] - one filter per channel
        // For depthwise conv, we need to apply each filter to its corresponding channel

        let (_batch, channels, _seq_len) = x.dims3()?;
        let kernel_size = 7;
        // CAUSAL padding: all padding on the left, none on the right
        let left_padding = kernel_size - 1; // = 6

        // Pad the input (causal: left only)
        let x_padded = x.pad_with_zeros(2, left_padding, 0)?;

        // Manual depthwise convolution
        let mut output_slices = Vec::with_capacity(channels);

        for c in 0..channels {
            // Extract single channel: [batch, 1, seq_len + 2*padding]
            let x_c = x_padded.narrow(1, c, 1)?;

            // Get filter for this channel: [1, 1, kernel_size]
            let w_c = self.dwconv_weight.narrow(0, c, 1)?;

            // Apply convolution using candle's conv1d
            let cfg = Conv1dConfig {
                padding: 0,
                ..Default::default()
            };
            let conv = Conv1d::new(w_c, None, cfg);
            let out_c = conv.forward(&x_c)?;

            output_slices.push(out_c);
        }

        // Concatenate along channel dimension
        let output = Tensor::cat(&output_slices, 1)?;

        // Add bias: [channels] -> [1, channels, 1]
        let bias = self.dwconv_bias.unsqueeze(0)?.unsqueeze(2)?;
        output.broadcast_add(&bias)
    }
}

/// ConvNeXt Upsample Block.
///
/// Structure (from decoder.upsample.{0,1}):
/// - ConvTranspose1d for 2x upsampling (*.0.conv)
/// - ConvNeXtBlock (*.1)
#[derive(Debug)]
struct ConvNeXtUpsampleBlock {
    upsample_conv: ConvTranspose1d,
    convnext: ConvNeXtBlock,
}

impl ConvNeXtUpsampleBlock {
    fn from_vb(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        // ConvTranspose1d: [channels, channels, 2] stride=2 for 2x upsample
        let upsample_weight = vb.get((channels, channels, 2), "0.conv.weight")?;
        let upsample_bias = vb.get((channels,), "0.conv.bias")?;

        let cfg = ConvTranspose1dConfig {
            stride: 2,
            padding: 0,
            output_padding: 0,
            ..Default::default()
        };
        let upsample_conv = ConvTranspose1d::new(upsample_weight, Some(upsample_bias), cfg);

        // ConvNeXt block
        let convnext = ConvNeXtBlock::from_vb(vb.pp("1"), channels)?;

        Ok(Self {
            upsample_conv,
            convnext,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, channels, seq_len]

        // Debug: before upsample
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: upsample_input[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
        }

        // 2x upsample via ConvTranspose1d
        let x = self.upsample_conv.forward(x)?;

        // Debug: after ConvTranspose1d, before trim
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: after_transconv[:5, 0]: {:?}, shape={:?}",
                &first[..5.min(first.len())],
                x.dims()
            );
        }

        // Causal trimming for transposed convolution (matching Python CausalTransConvNet)
        // Python formula: pad = kernel_size - stride; left_pad = right_pad = ceil(pad)
        // For kernel=2, stride=2: pad = 0, so no trimming needed
        // But this code was using kernel_size-1 = 1, which was WRONG
        let seq_len = x.dims()[2];
        let kernel_size = 2usize;
        let stride = 2usize;
        let pad = kernel_size.saturating_sub(stride); // = 0 for kernel=2, stride=2
        let x = if pad > 0 {
            x.narrow(2, pad, seq_len - 2 * pad)?
        } else {
            x // No trimming needed
        };

        // Debug: after trim, before ConvNeXt
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: after_trim[:5, 0]: {:?}, shape={:?}",
                &first[..5.min(first.len())],
                x.dims()
            );
        }

        // ConvNeXt block
        self.convnext.forward(&x)
    }
}

/// HiFi-GAN Residual Block with Snake activation.
///
/// Structure from Qwen3-TTS-Tokenizer:
/// - Snake activation (act1)
/// - CausalConv1d with dilation
/// - Snake activation (act2)
/// - CausalConv1d (1x1)
/// - Residual connection
#[derive(Debug)]
struct HifiResBlock {
    act1: Snake,
    conv1: Conv1d,
    conv1_padding: usize, // Causal padding for conv1
    act2: Snake,
    conv2: Conv1d,
    conv2_padding: usize, // Causal padding for conv2
}

impl HifiResBlock {
    /// Load from VarBuilder (Qwen3 format).
    ///
    /// `dilation` should be 1, 3, or 9 for the three residual blocks in each HiFi-GAN stage.
    fn from_vb(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> CandleResult<Self> {
        let act1 = Snake::from_vb(vb.pp("act1"), channels)?;
        let act2 = Snake::from_vb(vb.pp("act2"), channels)?;

        // CausalConvNet padding: (kernel_size - 1) * dilation (left padding only)
        // For inference, we use symmetric padding in Conv1d and then trim
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let conv1_padding = effective_kernel - 1; // Causal: all padding on left

        let conv1_cfg = Conv1dConfig {
            padding: 0, // We'll apply causal padding manually
            dilation,
            ..Default::default()
        };
        let conv1 = conv1d(
            channels,
            channels,
            kernel_size,
            conv1_cfg,
            vb.pp("conv1.conv"),
        )?;

        // Second conv is 1x1 projection (no dilation needed)
        let conv2_padding = 0; // 1x1 conv needs no padding
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
            conv1_padding,
            act2,
            conv2,
            conv2_padding,
        })
    }

    /// Create with random initialization.
    fn new_random(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let act1 = Snake::new_random(channels, device)?;
        let act2 = Snake::new_random(channels, device)?;

        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let conv1_padding = effective_kernel - 1;

        let w1 = Tensor::randn(0.0f32, 0.02, (channels, channels, kernel_size), device)?;
        let b1 = Tensor::zeros((channels,), DType::F32, device)?;
        let conv1 = Conv1d::new(
            w1,
            Some(b1),
            Conv1dConfig {
                padding: 0,
                dilation,
                ..Default::default()
            },
        );

        let w2 = Tensor::randn(0.0f32, 0.02, (channels, channels, 1), device)?;
        let b2 = Tensor::zeros((channels,), DType::F32, device)?;
        let conv2 = Conv1d::new(w2, Some(b2), Conv1dConfig::default());
        let conv2_padding = 0;

        Ok(Self {
            act1,
            conv1,
            conv1_padding,
            act2,
            conv2,
            conv2_padding,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        // Debug: input
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: resblock input[:5,0]: {:?}",
                &first[..5.min(first.len())]
            );
        }

        // Apply Snake activation 1
        let x = self.act1.forward(x)?;

        // Debug: after act1
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: resblock after_act1[:5,0]: {:?}",
                &first[..5.min(first.len())]
            );
        }

        // Apply causal padding (left padding only) for conv1
        let x = if self.conv1_padding > 0 {
            x.pad_with_zeros(2, self.conv1_padding, 0)?
        } else {
            x
        };
        let x = self.conv1.forward(&x)?;

        // Debug: after conv1
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: resblock after_conv1[:5,0]: {:?}, shape={:?}",
                &first[..5.min(first.len())],
                x.dims()
            );
        }

        // Apply Snake activation 2
        let x = self.act2.forward(&x)?;

        // Apply causal padding for conv2 (usually 0 for 1x1 conv)
        let x = if self.conv2_padding > 0 {
            x.pad_with_zeros(2, self.conv2_padding, 0)?
        } else {
            x
        };
        let x = self.conv2.forward(&x)?;

        // Debug: after conv2
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: resblock after_conv2[:5,0]: {:?}",
                &first[..5.min(first.len())]
            );
        }

        // Residual connection - ensure same length
        // Causal conv may produce different length, need to align
        let res_len = residual.dim(2)?;
        let out_len = x.dim(2)?;

        if out_len == res_len {
            x + residual
        } else if out_len > res_len {
            // Trim output to match residual (take FIRST res_len samples for causal)
            x.narrow(2, 0, res_len)? + residual
        } else {
            // This shouldn't happen with proper causal padding
            Err(candle_core::Error::Msg(format!(
                "Output length {} < residual length {}",
                out_len, res_len
            )))?
        }
    }
}

/// HiFi-GAN Upsample Block with Snake activation.
///
/// Structure from Qwen3-TTS-Tokenizer (decoder.decoder.{1-4}):
/// - Snake activation (block.0)
/// - CausalConvTranspose1d for upsampling (block.1.conv) with trimming
/// - 3 residual blocks (block.2, block.3, block.4)
#[derive(Debug)]
struct HifiUpsampleBlock {
    snake: Snake,
    upsample_conv: ConvTranspose1d,
    /// Trimming for causal conv transpose: (left_pad, right_pad)
    upsample_trim: (usize, usize),
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

        // Upsample conv (transposed convolution) - CausalConvTranspose style
        // Python: pad = kernel_size - stride; left_pad = ceil(pad); right_pad = left_pad
        // We use no padding in conv and manually trim the output
        let conv_cfg = ConvTranspose1dConfig {
            stride: upsample_factor,
            padding: 0, // No padding, we trim manually
            ..Default::default()
        };

        // Calculate trim amounts (matching Python CausalTransConvNet exactly)
        // Python formula:
        //   pad = kernel_size - stride
        //   left_pad = ceil(pad)
        //   right_pad = left_pad
        let pad = kernel_size.saturating_sub(upsample_factor);
        let trim_left = pad; // Already integer, ceil is identity
        let trim_right = pad;
        let upsample_trim = (trim_left, trim_right);

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

        // Residual blocks with dilations (1, 3, 9)
        let dilations = [1usize, 3, 9];
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for i in 0..num_res_blocks {
            let dilation = dilations.get(i).copied().unwrap_or(1);
            let block =
                HifiResBlock::from_vb(vb.pp(format!("block.{}", i + 2)), out_channels, 7, dilation)
                    .or_else(|_| HifiResBlock::new_random(out_channels, 7, dilation, device))?;
            res_blocks.push(block);
        }

        Ok(Self {
            snake,
            upsample_conv,
            upsample_trim,
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

        // Calculate trim amounts (matching Python CausalTransConvNet exactly)
        // Python formula:
        //   pad = kernel_size - stride
        //   left_pad = ceil(pad)
        //   right_pad = left_pad
        let pad = kernel_size.saturating_sub(upsample_factor);
        let trim_left = pad;
        let trim_right = pad;
        let upsample_trim = (trim_left, trim_right);

        let w = Tensor::randn(
            0.0f32,
            0.02,
            (in_channels, out_channels, kernel_size),
            device,
        )?;
        let b = Tensor::zeros((out_channels,), DType::F32, device)?;
        let conv_cfg = ConvTranspose1dConfig {
            stride: upsample_factor,
            padding: 0,
            ..Default::default()
        };
        let upsample_conv = ConvTranspose1d::new(w, Some(b), conv_cfg);

        let dilations = [1usize, 3, 9];
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for i in 0..num_res_blocks {
            let dilation = dilations.get(i).copied().unwrap_or(1);
            res_blocks.push(HifiResBlock::new_random(out_channels, 7, dilation, device)?);
        }

        Ok(Self {
            snake,
            upsample_conv,
            upsample_trim,
            res_blocks,
        })
    }

    #[allow(dead_code)]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Debug: before snake
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: hifi_before_snake[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
        }
        let x = self.snake.forward(x)?;
        // Debug: after snake
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: hifi_after_snake[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
        }
        let mut x = self.upsample_conv.forward(&x)?;

        // Apply causal trimming (matching Python CausalTransConvNet)
        let (left_trim, right_trim) = self.upsample_trim;
        if left_trim > 0 || right_trim > 0 {
            let len = x.dim(2)?;
            let start = left_trim;
            let new_len = len.saturating_sub(left_trim + right_trim);
            if new_len > 0 {
                x = x.narrow(2, start, new_len)?;
            }
        }

        // Debug: after transconv+trim
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: hifi_transconv[:5, 0]: {:?}, shape={:?}",
                &first[..5.min(first.len())],
                x.dims()
            );
        }

        for block in &self.res_blocks {
            x = block.forward(&x)?;
        }

        // Debug: after resblocks
        {
            let first: Vec<f32> = x.i((0, .., 0))?.to_vec1()?;
            debug_print!(
                "DEBUG: hifi_resblocks[:5, 0]: {:?}",
                &first[..5.min(first.len())]
            );
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
