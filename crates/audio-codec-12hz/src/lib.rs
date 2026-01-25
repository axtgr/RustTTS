//! # audio-codec-12hz
//!
//! Audio codec decoder for Qwen3-TTS-Tokenizer-12Hz.
//!
//! This crate provides decoding of acoustic tokens to PCM audio,
//! including:
//! - Neural codec decoder
//! - Streaming chunk output
//! - Overlap-add with Hann window crossfade
//! - WAV export

pub mod crossfade;
pub mod decoder;
pub mod wav;

use tts_core::{AudioChunk, AudioCodec, TtsError, TtsResult};

/// Default sample rate for the codec.
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// Default samples per acoustic token.
pub const SAMPLES_PER_TOKEN: usize = 2000;

/// Codec decoder for Qwen3-TTS-Tokenizer-12Hz.
///
/// Note: This is a placeholder. Full implementation in Phase 3.
#[derive(Debug)]
pub struct Codec12Hz {
    sample_rate: u32,
    samples_per_token: usize,
}

impl Default for Codec12Hz {
    fn default() -> Self {
        Self::new()
    }
}

impl Codec12Hz {
    /// Create a new codec decoder with default settings.
    pub fn new() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            samples_per_token: SAMPLES_PER_TOKEN,
        }
    }

    /// Create a codec with custom settings.
    pub fn with_settings(sample_rate: u32, samples_per_token: usize) -> Self {
        Self {
            sample_rate,
            samples_per_token,
        }
    }

    /// Load codec weights from a safetensors file.
    ///
    /// Note: Placeholder for Phase 3 implementation.
    pub fn load_weights(&mut self, _path: &std::path::Path) -> TtsResult<()> {
        Err(TtsError::internal("codec loading not yet implemented"))
    }
}

impl AudioCodec for Codec12Hz {
    fn decode(&self, tokens: &[u32]) -> TtsResult<AudioChunk> {
        if tokens.is_empty() {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        // Placeholder: generate silence for now
        // Real implementation will decode tokens through neural network
        let num_samples = tokens.len() * self.samples_per_token;
        let pcm = vec![0.0f32; num_samples];

        let duration_ms = (num_samples as f32 / self.sample_rate as f32) * 1000.0;

        Ok(AudioChunk::new(pcm, self.sample_rate, 0.0, duration_ms))
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn samples_per_token(&self) -> usize {
        self.samples_per_token
    }

    fn frame_hop(&self) -> usize {
        self.samples_per_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_creation() {
        let codec = Codec12Hz::new();
        assert_eq!(codec.sample_rate(), DEFAULT_SAMPLE_RATE);
        assert_eq!(codec.samples_per_token(), SAMPLES_PER_TOKEN);
    }

    #[test]
    fn test_decode_placeholder() {
        let codec = Codec12Hz::new();
        let tokens = vec![1, 2, 3, 4, 5];
        let chunk = codec.decode(&tokens).unwrap();

        assert_eq!(chunk.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(chunk.num_samples(), tokens.len() * SAMPLES_PER_TOKEN);
    }

    #[test]
    fn test_decode_empty_error() {
        let codec = Codec12Hz::new();
        let result = codec.decode(&[]);
        assert!(result.is_err());
    }
}
