//! Acoustic model configuration.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for the acoustic model (Talker).
///
/// Based on Qwen3-TTS-12Hz architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticModelConfig {
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Text embedding dimension (before projection).
    /// In Qwen3-TTS this is 2048, which gets projected to hidden_size (1024).
    pub embedding_dim: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Intermediate (MLP) dimension size.
    pub intermediate_size: usize,
    /// Head dimension (hidden_size / num_attention_heads).
    pub head_dim: usize,

    // Vocabulary
    /// Text vocabulary size.
    pub text_vocab_size: usize,
    /// Total codec vocabulary size (includes special tokens).
    pub codec_vocab_size: usize,
    /// Number of parallel codebook groups.
    pub num_code_groups: usize,
    /// Codebook size per group.
    pub codebook_size: usize,

    // Position embeddings
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// Rope theta for rotary embeddings.
    pub rope_theta: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,

    // Multimodal RoPE (mrope) settings
    /// Whether to use interleaved multimodal RoPE.
    pub mrope_interleaved: bool,
    /// Multimodal RoPE section sizes [temporal, height, width].
    /// Each section corresponds to a different position_id component.
    /// Sum of sections should equal head_dim / 2 (half rotation dimension).
    pub mrope_section: Vec<usize>,

    // Special tokens
    /// TTS BOS token ID in text vocabulary.
    pub tts_bos_token_id: u32,
    /// TTS EOS token ID in text vocabulary.
    pub tts_eos_token_id: u32,
    /// TTS PAD token ID in text vocabulary.
    pub tts_pad_token_id: u32,
    /// Codec BOS token ID in codec vocabulary.
    pub codec_bos_id: u32,
    /// Codec EOS token ID in codec vocabulary.
    pub codec_eos_id: u32,
    /// Codec PAD token ID in codec vocabulary.
    pub codec_pad_id: u32,
}

impl Default for AcousticModelConfig {
    /// Default configuration for Qwen3-TTS-12Hz-0.6B-Base.
    fn default() -> Self {
        Self::qwen3_tts_0_6b()
    }
}

impl AcousticModelConfig {
    /// Load configuration from a JSON file (HuggingFace config.json format).
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config file: {e}"))?;
        Self::from_json(&content)
    }

    /// Load configuration from a pretrained model directory.
    ///
    /// Looks for `config.json` in the directory.
    pub fn from_pretrained(dir: impl AsRef<Path>) -> Result<Self, String> {
        let config_path = dir.as_ref().join("config.json");
        if !config_path.exists() {
            return Err(format!(
                "config.json not found in {}",
                dir.as_ref().display()
            ));
        }
        Self::from_json_file(&config_path)
    }

    /// Parse configuration from JSON string (HuggingFace config.json format).
    ///
    /// Maps HuggingFace field names to our config structure.
    /// Supports both flat configs and Qwen3-TTS nested configs (with talker_config).
    pub fn from_json(json: &str) -> Result<Self, String> {
        let v: Value =
            serde_json::from_str(json).map_err(|e| format!("failed to parse JSON: {e}"))?;

        // Check if this is a nested Qwen3-TTS config with talker_config
        let talker = v.get("talker_config");

        // Helper to extract values with defaults, checking talker_config first
        let get_usize = |key: &str, default: usize| -> usize {
            // First try talker_config, then root
            talker
                .and_then(|t| t.get(key))
                .or_else(|| v.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let get_f64 = |key: &str, default: f64| -> f64 {
            talker
                .and_then(|t| t.get(key))
                .or_else(|| v.get(key))
                .and_then(|v| v.as_f64())
                .unwrap_or(default)
        };

        let get_u32 = |key: &str, default: u32| -> u32 {
            talker
                .and_then(|t| t.get(key))
                .or_else(|| v.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default)
        };

        // Map HuggingFace field names to our config
        // Qwen3-TTS uses: talker_config.hidden_size, talker_config.num_hidden_layers, etc.
        let hidden_size = get_usize("hidden_size", 1024);
        // Qwen3-TTS uses text_hidden_size=2048 which gets projected to hidden_size=1024
        let embedding_dim = get_usize("text_hidden_size", 2048);
        let num_attention_heads = get_usize("num_attention_heads", 16);
        let num_kv_heads = get_usize("num_key_value_heads", 8);
        let num_layers = get_usize("num_hidden_layers", 28);
        let intermediate_size = get_usize("intermediate_size", 3072);
        let head_dim = get_usize("head_dim", hidden_size / num_attention_heads);

        // Vocabulary sizes - use text_vocab_size for text, vocab_size for codec
        let text_vocab_size = get_usize("text_vocab_size", 151936);
        let codec_vocab_size = get_usize("vocab_size", 3072);
        let num_code_groups = get_usize("num_code_groups", 16);
        let codebook_size = get_usize("codebook_size", 2048);

        let max_position_embeddings = get_usize("max_position_embeddings", 32768);
        let rope_theta = get_f64("rope_theta", 1_000_000.0);
        let rms_norm_eps = get_f64("rms_norm_eps", 1e-6);

        // Parse rope_scaling for multimodal RoPE
        let rope_scaling = talker
            .and_then(|t| t.get("rope_scaling"))
            .or_else(|| v.get("rope_scaling"));

        let (mrope_interleaved, mrope_section) = if let Some(rs) = rope_scaling {
            let interleaved = rs
                .get("interleaved")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let section = rs
                .get("mrope_section")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            (interleaved, section)
        } else {
            (false, vec![])
        };

        // Special tokens - from root level in Qwen3-TTS
        let tts_bos_token_id = v
            .get("tts_bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(151672);
        let tts_eos_token_id = v
            .get("tts_eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(151673);
        let tts_pad_token_id = v
            .get("tts_pad_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(151671);
        let codec_bos_id = get_u32("codec_bos_id", 2149);
        let codec_eos_id = get_u32("codec_eos_token_id", 2150);
        let codec_pad_id = get_u32("codec_pad_id", 2148);

        Ok(Self {
            hidden_size,
            embedding_dim,
            num_attention_heads,
            num_kv_heads,
            num_layers,
            intermediate_size,
            head_dim,
            text_vocab_size,
            codec_vocab_size,
            num_code_groups,
            codebook_size,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            mrope_interleaved,
            mrope_section,
            tts_bos_token_id,
            tts_eos_token_id,
            tts_pad_token_id,
            codec_bos_id,
            codec_eos_id,
            codec_pad_id,
        })
    }

    /// Configuration for Qwen3-TTS-12Hz-0.6B-Base model.
    pub fn qwen3_tts_0_6b() -> Self {
        Self {
            // Talker dimensions
            hidden_size: 1024,
            embedding_dim: 2048, // text_embedding is [vocab_size, 2048]
            num_attention_heads: 16,
            num_kv_heads: 8,
            num_layers: 28,
            intermediate_size: 3072,
            // IMPORTANT: head_dim is explicitly 128 in config.json, NOT hidden_size/num_heads!
            // This means Q projection is [num_heads * head_dim, hidden_size] = [2048, 1024]
            head_dim: 128,

            // Vocabulary
            text_vocab_size: 151936,
            codec_vocab_size: 3072,
            num_code_groups: 16,
            codebook_size: 2048,

            // Position embeddings
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,

            // Multimodal RoPE (from config.json rope_scaling)
            // mrope_section sum = 64 = head_dim / 2 (for half rotation dimension)
            mrope_interleaved: true,
            mrope_section: vec![24, 20, 20], // temporal, height, width

            // Special tokens (from config.json)
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            codec_bos_id: 2149,
            codec_eos_id: 2150,
            codec_pad_id: 2148,
        }
    }

    /// Configuration for Qwen3-TTS-12Hz-1.7B-Base model.
    pub fn qwen3_tts_1_7b() -> Self {
        Self {
            // Talker dimensions (estimated, needs verification)
            hidden_size: 2048,
            embedding_dim: 4096, // estimated
            num_attention_heads: 16,
            num_kv_heads: 8,
            num_layers: 28,
            intermediate_size: 5504,
            head_dim: 128,

            // Vocabulary (same as 0.6B)
            text_vocab_size: 151936,
            codec_vocab_size: 3072,
            num_code_groups: 16,
            codebook_size: 2048,

            // Position embeddings
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,

            // Multimodal RoPE (estimated, needs verification)
            mrope_interleaved: true,
            mrope_section: vec![48, 40, 40], // estimated for head_dim=128

            // Special tokens (same as 0.6B)
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            codec_bos_id: 2149,
            codec_eos_id: 2150,
            codec_pad_id: 2148,
        }
    }

    /// Small configuration for testing.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            embedding_dim: 64, // same as hidden_size for testing
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_size: 128,
            head_dim: 16,

            text_vocab_size: 100,
            codec_vocab_size: 100,
            num_code_groups: 4,
            codebook_size: 20,

            max_position_embeddings: 256,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,

            // No multimodal RoPE for tiny config
            mrope_interleaved: false,
            mrope_section: vec![],

            tts_bos_token_id: 1,
            tts_eos_token_id: 2,
            tts_pad_token_id: 0,
            codec_bos_id: 1,
            codec_eos_id: 2,
            codec_pad_id: 0,
        }
    }

    /// Legacy configuration for backward compatibility.
    #[deprecated(
        since = "0.2.0",
        note = "Use qwen3_tts_0_6b() or qwen3_tts_1_7b() instead"
    )]
    pub fn legacy() -> Self {
        Self {
            hidden_size: 2048,
            embedding_dim: 2048,
            num_attention_heads: 16,
            num_kv_heads: 4,
            num_layers: 24,
            intermediate_size: 5504,
            head_dim: 128,

            text_vocab_size: 151936,
            codec_vocab_size: 65536,
            num_code_groups: 1,
            codebook_size: 65536,

            max_position_embeddings: 8192,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,

            // No multimodal RoPE for legacy
            mrope_interleaved: false,
            mrope_section: vec![],

            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            codec_bos_id: 0,
            codec_eos_id: 1,
            codec_pad_id: 2,
        }
    }
}

/// Configuration for the Code Predictor model.
///
/// Used for expanding delay pattern tokens to full multi-codebook output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePredictorConfig {
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
    /// Head dimension.
    pub head_dim: usize,
    /// Number of parallel codebook groups.
    pub num_code_groups: usize,
    /// Codebook size per group.
    pub codebook_size: usize,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// RoPE theta.
    pub rope_theta: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self::qwen3_tts_0_6b()
    }
}

impl CodePredictorConfig {
    /// Configuration for Qwen3-TTS-12Hz-0.6B code predictor.
    pub fn qwen3_tts_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_kv_heads: 8,
            num_layers: 5,
            intermediate_size: 3072,
            head_dim: 128,
            num_code_groups: 16,
            codebook_size: 2048,
            max_position_embeddings: 65536,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    /// Small configuration for testing.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_size: 128,
            head_dim: 16,
            num_code_groups: 4,
            codebook_size: 20,
            max_position_embeddings: 256,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AcousticModelConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_code_groups, 16);
    }

    #[test]
    fn test_qwen3_tts_0_6b_config() {
        let config = AcousticModelConfig::qwen3_tts_0_6b();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.head_dim, 128); // Explicitly 128, NOT hidden_size/num_heads!
        assert_eq!(config.text_vocab_size, 151936);
        assert_eq!(config.codec_vocab_size, 3072);
        assert_eq!(config.num_code_groups, 16);
        assert_eq!(config.codebook_size, 2048);
        // Multimodal RoPE: section sum = head_dim / 2 (half for rotation)
        assert!(config.mrope_interleaved);
        assert_eq!(config.mrope_section, vec![24, 20, 20]);
        assert_eq!(
            config.mrope_section.iter().sum::<usize>(),
            config.head_dim / 2
        );
    }

    #[test]
    fn test_code_predictor_config() {
        let config = CodePredictorConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 5);
        assert_eq!(config.num_code_groups, 16);
    }

    #[test]
    fn test_tiny_config() {
        let config = AcousticModelConfig::tiny();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_special_tokens() {
        let config = AcousticModelConfig::qwen3_tts_0_6b();
        assert_eq!(config.tts_bos_token_id, 151672);
        assert_eq!(config.tts_eos_token_id, 151673);
        assert_eq!(config.codec_bos_id, 2149);
        assert_eq!(config.codec_eos_id, 2150);
    }

    #[test]
    fn test_from_json() {
        // Test with Qwen3-TTS style config (nested talker_config)
        let json = r#"{
            "tts_bos_token_id": 100,
            "tts_eos_token_id": 101,
            "tts_pad_token_id": 0,
            "talker_config": {
                "hidden_size": 512,
                "text_hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 12,
                "intermediate_size": 2048,
                "text_vocab_size": 50000,
                "vocab_size": 1024,
                "num_code_groups": 8,
                "codebook_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 100000.0,
                "rms_norm_eps": 1e-5,
                "rope_scaling": {
                    "interleaved": true,
                    "mrope_section": [24, 20, 20]
                },
                "codec_bos_id": 10,
                "codec_eos_token_id": 11,
                "codec_pad_id": 0
            }
        }"#;

        let config = AcousticModelConfig::from_json(json).unwrap();

        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.embedding_dim, 1024);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.intermediate_size, 2048);
        assert_eq!(config.text_vocab_size, 50000);
        assert_eq!(config.codec_vocab_size, 1024);
        assert_eq!(config.num_code_groups, 8);
        assert_eq!(config.codebook_size, 512);
        assert_eq!(config.max_position_embeddings, 4096);
        assert!((config.rope_theta - 100000.0).abs() < 1e-6);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        // Multimodal RoPE
        assert!(config.mrope_interleaved);
        assert_eq!(config.mrope_section, vec![24, 20, 20]);
        assert_eq!(config.tts_bos_token_id, 100);
        assert_eq!(config.tts_eos_token_id, 101);
        assert_eq!(config.codec_bos_id, 10);
        assert_eq!(config.codec_eos_id, 11);
    }

    #[test]
    fn test_from_json_with_defaults() {
        // Minimal JSON - should use defaults for missing fields
        let json = r#"{
            "hidden_size": 256,
            "num_hidden_layers": 4
        }"#;

        let config = AcousticModelConfig::from_json(json).unwrap();

        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 4);
        // Check defaults
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.codec_vocab_size, 3072);
        assert_eq!(config.tts_bos_token_id, 151672);
        // No multimodal RoPE without rope_scaling
        assert!(!config.mrope_interleaved);
        assert!(config.mrope_section.is_empty());
    }

    #[test]
    fn test_from_json_invalid() {
        let json = "not valid json";
        let result = AcousticModelConfig::from_json(json);
        assert!(result.is_err());
    }
}
