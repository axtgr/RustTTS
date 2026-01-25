//! Configuration structures for the TTS engine.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier.
    pub name: String,
    /// Model version.
    pub version: String,

    /// Transformer architecture settings.
    pub transformer: TransformerConfig,

    /// Codec settings.
    pub codec: CodecConfig,

    /// Tokenizer settings.
    pub tokenizer: TokenizerConfig,

    /// Path to model weights (safetensors).
    pub weights_path: PathBuf,

    /// Compute device preference.
    #[serde(default)]
    pub device: DeviceConfig,
}

/// Transformer architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Intermediate (MLP) dimension size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Rope theta for rotary embeddings.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10000.0
}

/// Audio codec configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Sample rate in Hz.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// Number of samples per codec frame.
    #[serde(default = "default_frame_size")]
    pub frame_size: usize,
    /// Frame hop size in samples.
    #[serde(default = "default_frame_hop")]
    pub frame_hop: usize,
    /// Number of codec codebook entries.
    pub codebook_size: usize,
    /// Number of residual vector quantization layers.
    #[serde(default = "default_num_rvq")]
    pub num_rvq: usize,
}

fn default_sample_rate() -> u32 {
    24000
}

fn default_frame_size() -> usize {
    2000
}

fn default_frame_hop() -> usize {
    2000
}

fn default_num_rvq() -> usize {
    1
}

/// Tokenizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Path to tokenizer model file.
    pub model_path: PathBuf,
    /// BOS token.
    pub bos_token: Option<String>,
    /// EOS token.
    pub eos_token: Option<String>,
    /// PAD token.
    pub pad_token: Option<String>,
}

/// Compute device configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Preferred device type.
    #[serde(default)]
    pub device_type: DeviceType,
    /// Specific GPU device index (if using CUDA).
    pub gpu_index: Option<usize>,
    /// Data type for inference.
    #[serde(default)]
    pub dtype: DType,
}

/// Device type for computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    /// CPU computation.
    #[default]
    Cpu,
    /// CUDA GPU computation.
    Cuda,
    /// Metal GPU computation (Apple).
    Metal,
}

/// Data type for tensor computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// 32-bit floating point.
    #[default]
    F32,
    /// 16-bit floating point.
    F16,
    /// Brain floating point 16.
    Bf16,
}

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    /// Batching configuration.
    #[serde(default)]
    pub batching: BatchingConfig,

    /// Queue configuration.
    #[serde(default)]
    pub queue: QueueConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Metrics configuration.
    #[serde(default)]
    pub metrics: MetricsConfig,
}

/// Batching configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Maximum batch size.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    /// Batch window duration in milliseconds.
    #[serde(default = "default_batch_window_ms")]
    pub batch_window_ms: u64,
    /// Maximum tokens per batch.
    #[serde(default = "default_max_batch_tokens")]
    pub max_batch_tokens: usize,
}

fn default_max_batch_size() -> usize {
    8
}

fn default_batch_window_ms() -> u64 {
    10
}

fn default_max_batch_tokens() -> usize {
    4096
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: default_max_batch_size(),
            batch_window_ms: default_batch_window_ms(),
            max_batch_tokens: default_max_batch_tokens(),
        }
    }
}

/// Queue configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum queue size.
    #[serde(default = "default_max_queue_size")]
    pub max_queue_size: usize,
    /// Request timeout in milliseconds.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
}

fn default_max_queue_size() -> usize {
    1000
}

fn default_request_timeout_ms() -> u64 {
    30000
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: default_max_queue_size(),
            request_timeout_ms: default_request_timeout_ms(),
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level.
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Output format (json or text).
    #[serde(default = "default_log_format")]
    pub format: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "json".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
        }
    }
}

/// Metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection.
    #[serde(default = "default_metrics_enabled")]
    pub enabled: bool,
    /// Prometheus exporter port (if enabled).
    #[serde(default = "default_metrics_port")]
    pub port: u16,
}

fn default_metrics_enabled() -> bool {
    true
}

fn default_metrics_port() -> u16 {
    9090
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: default_metrics_enabled(),
            port: default_metrics_port(),
        }
    }
}

/// Server configuration (for tts-server).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server bind address.
    #[serde(default = "default_server_host")]
    pub host: String,
    /// Server port.
    #[serde(default = "default_server_port")]
    pub port: u16,
    /// Maximum concurrent connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    /// Request body size limit in bytes.
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
    /// Enable TLS.
    #[serde(default)]
    pub tls: Option<TlsConfig>,
    /// Authentication configuration.
    #[serde(default)]
    pub auth: Option<AuthConfig>,
}

fn default_server_host() -> String {
    "0.0.0.0".to_string()
}

fn default_server_port() -> u16 {
    8080
}

fn default_max_connections() -> usize {
    1000
}

fn default_max_body_size() -> usize {
    10 * 1024 * 1024 // 10 MB
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_server_host(),
            port: default_server_port(),
            max_connections: default_max_connections(),
            max_body_size: default_max_body_size(),
            tls: None,
            auth: None,
        }
    }
}

/// TLS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to certificate file.
    pub cert_path: PathBuf,
    /// Path to private key file.
    pub key_path: PathBuf,
}

/// Authentication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable API key authentication.
    #[serde(default)]
    pub api_key_enabled: bool,
    /// API key header name.
    #[serde(default = "default_api_key_header")]
    pub api_key_header: String,
}

fn default_api_key_header() -> String {
    "X-API-Key".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        assert_eq!(config.device_type, DeviceType::Cpu);
        assert!(config.gpu_index.is_none());
        assert_eq!(config.dtype, DType::F32);
    }

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.batching.max_batch_size, 8);
        assert_eq!(config.queue.max_queue_size, 1000);
        assert_eq!(config.logging.level, "info");
        assert!(config.metrics.enabled);
    }

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert!(config.tls.is_none());
        assert!(config.auth.is_none());
    }
}
