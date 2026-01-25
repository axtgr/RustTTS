//! # acoustic-model
//!
//! Acoustic model (transformer) implementation for Qwen3-TTS.
//!
//! This crate provides the core transformer model for converting text tokens
//! to acoustic tokens, including:
//! - Transformer architecture with KV cache
//! - Rotary position embeddings (RoPE)
//! - Streaming autoregressive generation
//! - Multiple sampling strategies (greedy, top-k, top-p)
//!
//! # Architecture
//!
//! The model follows the Qwen3-TTS architecture:
//! - Embedding layer
//! - N transformer blocks with:
//!   - RMSNorm (pre-normalization)
//!   - Multi-head attention with GQA (grouped query attention)
//!   - MLP with SwiGLU activation
//! - Output projection to acoustic token vocabulary
//!
//! # Example
//!
//! ```ignore
//! use acoustic_model::{Model, config::AcousticModelConfig, sampling::SamplingConfig};
//! use candle_core::Device;
//!
//! let config = AcousticModelConfig::default();
//! let device = Device::Cpu;
//! let model = Model::load("model.safetensors", config, &device)?;
//!
//! let input_tokens = vec![1, 2, 3, 4];
//! let sampling = SamplingConfig::default();
//! let output = model.generate(&input_tokens, 100, sampling, Some(2))?;
//! ```

pub mod cache;
pub mod config;
pub mod layers;
pub mod model;
pub mod sampling;

// Re-exports for convenience
pub use cache::{CacheEntry, CacheHandle, KvCacheManager};
pub use config::AcousticModelConfig;
pub use layers::{Attention, MLP, RmsNorm, RotaryEmbedding, TransformerBlock};
pub use model::{Model, StreamingGenerator};
pub use sampling::{Sampler, SamplingConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        // Default is now Qwen3-TTS-12Hz-0.6B-Base
        let config = AcousticModelConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_code_groups, 16);
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
    }

    #[test]
    fn test_cache_manager() {
        let manager = KvCacheManager::new(10, 64, 4, 256);
        assert!(manager.is_empty());
    }
}
