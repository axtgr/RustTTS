//! Trait definitions for TTS pipeline components.

use crate::error::TtsResult;
use crate::types::{AudioChunk, Lang, NormText, TokenSeq};

/// Text normalization trait.
///
/// Implementations convert raw input text into a normalized form suitable
/// for tokenization, handling numbers, dates, abbreviations, etc.
pub trait TextNormalizer: Send + Sync {
    /// Normalize the input text.
    ///
    /// # Arguments
    /// * `input` - Raw input text
    /// * `lang_hint` - Optional language hint for ambiguous text
    ///
    /// # Returns
    /// Normalized text with language and span information.
    fn normalize(&self, input: &str, lang_hint: Option<Lang>) -> TtsResult<NormText>;
}

/// Text tokenization trait.
///
/// Implementations convert normalized text into token sequences compatible
/// with the acoustic model.
pub trait TextTokenizer: Send + Sync {
    /// Encode normalized text into tokens.
    ///
    /// # Arguments
    /// * `text` - Normalized text to tokenize
    ///
    /// # Returns
    /// Token sequence with IDs and offset mappings.
    fn encode(&self, text: &NormText) -> TtsResult<TokenSeq>;

    /// Decode tokens back to text (for debugging).
    ///
    /// # Arguments
    /// * `tokens` - Token sequence to decode
    ///
    /// # Returns
    /// Decoded text string.
    fn decode(&self, tokens: &TokenSeq) -> TtsResult<String>;

    /// Get the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Get the BOS (beginning of sequence) token ID.
    fn bos_token_id(&self) -> Option<u32>;

    /// Get the EOS (end of sequence) token ID.
    fn eos_token_id(&self) -> Option<u32>;

    /// Get the PAD token ID.
    fn pad_token_id(&self) -> Option<u32>;
}

/// Audio codec trait for decoding acoustic tokens to PCM.
///
/// Implementations decode the output of the acoustic model into
/// audio waveforms.
pub trait AudioCodec: Send + Sync {
    /// Decode acoustic tokens into an audio chunk.
    ///
    /// # Arguments
    /// * `tokens` - Acoustic tokens to decode
    ///
    /// # Returns
    /// Decoded audio chunk.
    fn decode(&self, tokens: &[u32]) -> TtsResult<AudioChunk>;

    /// Get the codec's sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Get the number of samples per acoustic token.
    fn samples_per_token(&self) -> usize;

    /// Get the frame hop size in samples.
    fn frame_hop(&self) -> usize;
}

/// Generation options for the acoustic model.
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (1.0 = no change).
    pub temperature: f32,
    /// Top-k sampling parameter (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled).
    pub top_p: f32,
    /// Random seed for deterministic generation.
    pub seed: Option<u64>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}

impl GenerationOptions {
    /// Create new generation options with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k sampling.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p sampling.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_options_builder() {
        let opts = GenerationOptions::new()
            .with_max_tokens(1000)
            .with_temperature(0.8)
            .with_top_k(50)
            .with_top_p(0.9)
            .with_seed(42);

        assert_eq!(opts.max_tokens, 1000);
        assert!((opts.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(opts.top_k, 50);
        assert!((opts.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(opts.seed, Some(42));
    }

    #[test]
    fn test_generation_options_default() {
        let opts = GenerationOptions::default();
        assert_eq!(opts.max_tokens, 2048);
        assert!((opts.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(opts.top_k, 0);
        assert!((opts.top_p - 1.0).abs() < f32::EPSILON);
        assert!(opts.seed.is_none());
    }
}
