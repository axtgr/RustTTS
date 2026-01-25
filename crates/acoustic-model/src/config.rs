//! Acoustic model configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the acoustic model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticModelConfig {
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
    /// Text vocabulary size.
    pub text_vocab_size: usize,
    /// Acoustic token vocabulary size.
    pub acoustic_vocab_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Rope theta for rotary embeddings.
    pub rope_theta: f32,
}

impl Default for AcousticModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_kv_heads: 4,
            num_layers: 24,
            intermediate_size: 5504,
            text_vocab_size: 151936,
            acoustic_vocab_size: 65536,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AcousticModelConfig::default();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
    }
}
