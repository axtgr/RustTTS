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
use tracing::{debug, info, instrument};

use tts_core::{TtsError, TtsResult};

/// Configuration for the codec decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Number of codebook entries.
    pub codebook_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of residual vector quantization layers.
    pub num_rvq: usize,
    /// Hidden dimension for decoder layers.
    pub hidden_dim: usize,
    /// Number of residual blocks.
    pub num_residual_blocks: usize,
    /// Upsampling factors for each stage.
    pub upsample_factors: Vec<usize>,
    /// Kernel sizes for upsampling convolutions.
    pub upsample_kernels: Vec<usize>,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            codebook_size: 65536,
            embed_dim: 512,
            num_rvq: 1,
            hidden_dim: 512,
            num_residual_blocks: 3,
            // Total upsample: 8 * 5 * 5 * 10 = 2000 (samples per token)
            upsample_factors: vec![8, 5, 5, 10],
            upsample_kernels: vec![16, 10, 10, 20],
        }
    }
}

impl DecoderConfig {
    /// Calculate total upsampling factor.
    pub fn total_upsample(&self) -> usize {
        self.upsample_factors.iter().product()
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
#[derive(Debug)]
pub struct NeuralDecoder {
    /// Codebook embedding.
    codebook: Tensor,
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
    pub fn new(config: DecoderConfig, device: &Device) -> CandleResult<Self> {
        // Initialize codebook with random values
        let codebook = Tensor::randn(
            0.0f32,
            0.02,
            (config.codebook_size, config.embed_dim),
            device,
        )?;

        // Create placeholder layers (will be replaced when loading weights)
        let vb = VarBuilder::zeros(DType::F32, device);

        let input_conv = conv1d(
            config.embed_dim,
            config.hidden_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )?;

        let mut upsample_blocks = Vec::new();
        let mut channels = config.hidden_dim;

        for (i, (&factor, &kernel)) in config
            .upsample_factors
            .iter()
            .zip(config.upsample_kernels.iter())
            .enumerate()
        {
            let out_channels = channels / 2;
            let block = UpsampleBlock::new(
                channels,
                out_channels.max(32),
                factor,
                kernel,
                config.num_residual_blocks,
                vb.pp(format!("upsample_{i}")),
            )?;
            upsample_blocks.push(block);
            channels = out_channels.max(32);
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
            codebook,
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
    pub fn from_vb(vb: VarBuilder, config: DecoderConfig, device: &Device) -> TtsResult<Self> {
        let codebook = vb
            .get((config.codebook_size, config.embed_dim), "codebook")
            .map_err(|e| TtsError::internal(format!("failed to load codebook: {e}")))?;

        let input_conv = conv1d(
            config.embed_dim,
            config.hidden_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )
        .map_err(|e| TtsError::internal(format!("failed to load input_conv: {e}")))?;

        let mut upsample_blocks = Vec::new();
        let mut channels = config.hidden_dim;

        for (i, (&factor, &kernel)) in config
            .upsample_factors
            .iter()
            .zip(config.upsample_kernels.iter())
            .enumerate()
        {
            let out_channels = (channels / 2).max(32);
            let block = UpsampleBlock::new(
                channels,
                out_channels,
                factor,
                kernel,
                config.num_residual_blocks,
                vb.pp(format!("upsample_{i}")),
            )
            .map_err(|e| TtsError::internal(format!("failed to load upsample_{i}: {e}")))?;

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
            "Decoder loaded: codebook={}, embed={}, upsample={}x",
            config.codebook_size,
            config.embed_dim,
            config.total_upsample()
        );

        Ok(Self {
            codebook,
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

    /// Decode acoustic tokens to audio samples.
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

        let result = self
            .decode_internal(tokens)
            .map_err(|e| TtsError::internal(format!("decode failed: {e}")))?;

        debug!(
            "Decoded {} tokens to {} samples",
            tokens.len(),
            result.len()
        );
        Ok(result)
    }

    fn decode_internal(&self, tokens: &[u32]) -> CandleResult<Vec<f32>> {
        // Convert tokens to tensor
        let token_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?;

        // Lookup embeddings from codebook
        let embeddings = self.codebook.index_select(&token_tensor, 0)?;

        // Reshape to [batch=1, channels, seq_len]
        let embeddings = embeddings.unsqueeze(0)?.transpose(1, 2)?;

        // Input projection
        let mut x = self.input_conv.forward(&embeddings)?;
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

    /// Decode a single token (for streaming).
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
        assert_eq!(config.codebook_size, 65536);
        assert_eq!(config.total_upsample(), 2000);
    }

    #[test]
    fn test_decoder_config_upsample() {
        let config = DecoderConfig {
            upsample_factors: vec![4, 4, 4, 4],
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

        assert_eq!(samples.len(), 3 * 2000);

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
        let config = DecoderConfig {
            codebook_size: 1000,
            embed_dim: 64,
            hidden_dim: 64,
            num_residual_blocks: 1,
            upsample_factors: vec![4, 4],
            upsample_kernels: vec![8, 8],
            ..Default::default()
        };

        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device);

        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.config().codebook_size, 1000);
    }

    #[test]
    fn test_neural_decoder_decode() {
        let config = DecoderConfig {
            codebook_size: 100,
            embed_dim: 32,
            hidden_dim: 32,
            num_residual_blocks: 1,
            upsample_factors: vec![2, 2],
            upsample_kernels: vec![4, 4],
            ..Default::default()
        };

        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config.clone(), &device).unwrap();

        let tokens = vec![1, 2, 3];
        let samples = decoder.decode(&tokens).unwrap();

        // Should produce tokens * upsample_factor samples
        let expected_len = tokens.len() * config.total_upsample();
        assert_eq!(samples.len(), expected_len);
    }

    #[test]
    fn test_neural_decoder_invalid_token() {
        let config = DecoderConfig {
            codebook_size: 100,
            embed_dim: 32,
            hidden_dim: 32,
            num_residual_blocks: 1,
            upsample_factors: vec![2],
            upsample_kernels: vec![4],
            ..Default::default()
        };

        let device = Device::Cpu;
        let decoder = NeuralDecoder::new(config, &device).unwrap();

        // Token 200 exceeds codebook size of 100
        let result = decoder.decode(&[200]);
        assert!(result.is_err());
    }
}
