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
//! - Multi-codebook prediction (MTP module)
//!
//! # Architecture
//!
//! The Qwen3-TTS-12Hz model uses a dual-track architecture:
//!
//! 1. **Talker (Main LM)**: Generates the zeroth (semantic) codebook tokens
//!    - 28 transformer layers (0.6B) or 28 layers (1.7B)
//!    - GQA with 16 heads, 8 KV heads
//!    - Hidden size 1024 (0.6B) or 2048 (1.7B)
//!
//! 2. **Code Predictor (MTP)**: Predicts residual codebooks 1-15 from zeroth
//!    - 5 transformer layers
//!    - Parallel prediction of all residual codebooks
//!    - Enables single-frame instant generation
//!
//! # Example
//!
//! ```ignore
//! use acoustic_model::{Model, CodePredictor, config::AcousticModelConfig, sampling::SamplingConfig};
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
pub mod code_predictor;
pub mod config;
pub mod layers;
pub mod model;
pub mod sampling;

// Re-exports for convenience
pub use cache::{CacheEntry, CacheHandle, KvCacheManager};
pub use code_predictor::{CodePredictor, MultiCodebookOutput};
pub use config::{AcousticModelConfig, CodePredictorConfig};
pub use layers::{Attention, MLP, RmsNorm, RotaryEmbedding, TextProjection, TransformerBlock};
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
