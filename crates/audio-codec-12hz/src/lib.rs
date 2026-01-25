//! # audio-codec-12hz
//!
//! Audio codec decoder for Qwen3-TTS-Tokenizer-12Hz.
//!
//! This crate provides decoding of acoustic tokens to PCM audio,
//! including:
//! - Neural codec decoder
//! - Streaming chunk output with overlap-add
//! - Hann window crossfade for seamless audio
//! - WAV export

pub mod crossfade;
pub mod decoder;
pub mod wav;

use std::path::Path;

use candle_core::Device;
use tracing::{debug, info, instrument};

use tts_core::{AudioChunk, AudioCodec, TtsError, TtsResult};

pub use crossfade::Crossfader;
pub use decoder::{DecoderConfig, MockDecoder, NeuralDecoder};

/// Default sample rate for the codec.
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// Default samples per acoustic token (1920 for Qwen3-TTS at 12.5 Hz).
/// At 24kHz sample rate: 24000 / 12.5 = 1920 samples per frame.
pub const SAMPLES_PER_TOKEN: usize = 1920;

/// Default crossfade duration in milliseconds.
pub const DEFAULT_CROSSFADE_MS: f32 = 5.0;

/// Decoder backend selection.
#[derive(Debug)]
pub enum DecoderBackend {
    /// Neural network decoder (requires weights).
    Neural(NeuralDecoder),
    /// Mock decoder for testing (generates sine waves).
    Mock(MockDecoder),
}

/// Codec decoder for Qwen3-TTS-Tokenizer-12Hz.
///
/// Supports both neural network decoding (with loaded weights)
/// and mock decoding for testing purposes.
#[derive(Debug)]
pub struct Codec12Hz {
    backend: DecoderBackend,
    sample_rate: u32,
    samples_per_token: usize,
}

impl Default for Codec12Hz {
    fn default() -> Self {
        Self::new_mock()
    }
}

impl Codec12Hz {
    /// Create a new codec with mock decoder (for testing).
    pub fn new_mock() -> Self {
        let config = DecoderConfig::default();
        let samples_per_token = config.total_upsample();
        let backend = DecoderBackend::Mock(MockDecoder::new(config, DEFAULT_SAMPLE_RATE));

        Self {
            backend,
            sample_rate: DEFAULT_SAMPLE_RATE,
            samples_per_token,
        }
    }

    /// Create a new codec with neural decoder (random weights, for testing).
    pub fn new_neural(device: &Device) -> TtsResult<Self> {
        let config = DecoderConfig::default();
        let samples_per_token = config.total_upsample();

        let decoder = NeuralDecoder::new(config, device)
            .map_err(|e| TtsError::internal(format!("failed to create decoder: {e}")))?;

        Ok(Self {
            backend: DecoderBackend::Neural(decoder),
            sample_rate: DEFAULT_SAMPLE_RATE,
            samples_per_token,
        })
    }

    /// Load codec from safetensors weights file.
    #[instrument(skip(device), fields(path = %path.as_ref().display()))]
    pub fn load(path: impl AsRef<Path>, device: &Device) -> TtsResult<Self> {
        let config = DecoderConfig::default();
        let samples_per_token = config.total_upsample();

        let decoder = NeuralDecoder::load(path, config, device)?;

        info!("Codec loaded successfully");

        Ok(Self {
            backend: DecoderBackend::Neural(decoder),
            sample_rate: DEFAULT_SAMPLE_RATE,
            samples_per_token,
        })
    }

    /// Load codec with custom configuration.
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: DecoderConfig,
        device: &Device,
    ) -> TtsResult<Self> {
        let samples_per_token = config.total_upsample();
        let decoder = NeuralDecoder::load(path, config, device)?;

        Ok(Self {
            backend: DecoderBackend::Neural(decoder),
            sample_rate: DEFAULT_SAMPLE_RATE,
            samples_per_token,
        })
    }

    /// Check if using neural backend.
    pub fn is_neural(&self) -> bool {
        matches!(self.backend, DecoderBackend::Neural(_))
    }

    /// Create a streaming decoder from this codec.
    pub fn into_streaming(self, crossfade_ms: f32) -> StreamingDecoder {
        StreamingDecoder::new(self, crossfade_ms)
    }
}

impl AudioCodec for Codec12Hz {
    fn decode(&self, tokens: &[u32]) -> TtsResult<AudioChunk> {
        if tokens.is_empty() {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        let pcm = match &self.backend {
            DecoderBackend::Neural(decoder) => decoder.decode(tokens)?,
            DecoderBackend::Mock(decoder) => decoder.decode(tokens)?,
        };

        let duration_ms = (pcm.len() as f32 / self.sample_rate as f32) * 1000.0;

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

/// Streaming decoder with overlap-add crossfade.
///
/// Processes tokens incrementally and produces seamless audio chunks
/// using Hann window crossfade for smooth transitions.
#[derive(Debug)]
pub struct StreamingDecoder {
    codec: Codec12Hz,
    crossfader: Crossfader,
    total_samples: usize,
    chunk_index: usize,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new(codec: Codec12Hz, crossfade_ms: f32) -> Self {
        let crossfader = Crossfader::new(crossfade_ms, codec.sample_rate());

        Self {
            codec,
            crossfader,
            total_samples: 0,
            chunk_index: 0,
        }
    }

    /// Create with default crossfade (5ms).
    pub fn with_default_crossfade(codec: Codec12Hz) -> Self {
        Self::new(codec, DEFAULT_CROSSFADE_MS)
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate()
    }

    /// Get the total samples produced so far.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get the number of chunks processed.
    pub fn chunk_count(&self) -> usize {
        self.chunk_index
    }

    /// Process a batch of tokens and return crossfaded audio chunk.
    #[instrument(skip(self, tokens), fields(num_tokens = tokens.len(), chunk_idx = self.chunk_index))]
    pub fn process(&mut self, tokens: &[u32]) -> TtsResult<AudioChunk> {
        if tokens.is_empty() {
            return Err(TtsError::invalid_input("empty token sequence"));
        }

        // Decode tokens to raw audio
        let raw_chunk = self.codec.decode(tokens)?;

        // Apply crossfade
        let crossfaded = self.crossfader.process(&raw_chunk.pcm);

        let start_ms = (self.total_samples as f32 / self.codec.sample_rate() as f32) * 1000.0;
        let duration_ms = (crossfaded.len() as f32 / self.codec.sample_rate() as f32) * 1000.0;

        self.total_samples += crossfaded.len();
        self.chunk_index += 1;

        debug!(
            chunk_idx = self.chunk_index,
            samples = crossfaded.len(),
            total_samples = self.total_samples,
            "Processed streaming chunk"
        );

        Ok(AudioChunk::new(
            crossfaded,
            self.codec.sample_rate(),
            start_ms,
            duration_ms,
        ))
    }

    /// Flush remaining audio in the crossfade buffer.
    pub fn flush(&mut self) -> TtsResult<AudioChunk> {
        let flushed = self.crossfader.flush();

        if flushed.is_empty() {
            return Err(TtsError::invalid_input("no audio to flush"));
        }

        let start_ms = (self.total_samples as f32 / self.codec.sample_rate() as f32) * 1000.0;
        let duration_ms = (flushed.len() as f32 / self.codec.sample_rate() as f32) * 1000.0;

        self.total_samples += flushed.len();

        debug!(
            samples = flushed.len(),
            total_samples = self.total_samples,
            "Flushed streaming buffer"
        );

        Ok(AudioChunk::new(
            flushed,
            self.codec.sample_rate(),
            start_ms,
            duration_ms,
        ))
    }

    /// Reset the streaming state.
    pub fn reset(&mut self) {
        self.crossfader.reset();
        self.total_samples = 0;
        self.chunk_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_mock_creation() {
        let codec = Codec12Hz::new_mock();
        assert_eq!(codec.sample_rate(), DEFAULT_SAMPLE_RATE);
        assert_eq!(codec.samples_per_token(), SAMPLES_PER_TOKEN);
        assert!(!codec.is_neural());
    }

    #[test]
    fn test_codec_decode_mock() {
        let codec = Codec12Hz::new_mock();
        let tokens = vec![100, 200, 300];
        let chunk = codec.decode(&tokens).unwrap();

        assert_eq!(chunk.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(chunk.num_samples(), tokens.len() * SAMPLES_PER_TOKEN);

        // Check samples are in valid range
        for &s in chunk.pcm.iter() {
            assert!((-1.0..=1.0).contains(&s), "Sample out of range: {}", s);
        }
    }

    #[test]
    fn test_codec_decode_empty_error() {
        let codec = Codec12Hz::new_mock();
        let result = codec.decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_codec_neural_creation() {
        let device = Device::Cpu;
        let codec = Codec12Hz::new_neural(&device).unwrap();
        assert!(codec.is_neural());
        assert_eq!(codec.sample_rate(), DEFAULT_SAMPLE_RATE);
    }

    #[test]
    fn test_streaming_decoder_creation() {
        let codec = Codec12Hz::new_mock();
        let streaming = StreamingDecoder::new(codec, 5.0);

        assert_eq!(streaming.sample_rate(), DEFAULT_SAMPLE_RATE);
        assert_eq!(streaming.total_samples(), 0);
        assert_eq!(streaming.chunk_count(), 0);
    }

    #[test]
    fn test_streaming_decoder_process() {
        let codec = Codec12Hz::new_mock();
        let mut streaming = StreamingDecoder::new(codec, 5.0);

        // Process first chunk
        let chunk1 = streaming.process(&[100, 101, 102]).unwrap();
        assert!(chunk1.num_samples() > 0);
        assert_eq!(streaming.chunk_count(), 1);

        // Process second chunk
        let chunk2 = streaming.process(&[200, 201, 202]).unwrap();
        assert!(chunk2.num_samples() > 0);
        assert_eq!(streaming.chunk_count(), 2);

        // Total samples should be accumulating
        assert!(streaming.total_samples() > 0);
    }

    #[test]
    fn test_streaming_decoder_flush() {
        let codec = Codec12Hz::new_mock();
        let mut streaming = StreamingDecoder::new(codec, 5.0);

        // Process a chunk
        streaming.process(&[100, 101, 102]).unwrap();

        // Flush should return remaining samples
        let flushed = streaming.flush().unwrap();
        assert!(flushed.num_samples() > 0);
    }

    #[test]
    fn test_streaming_decoder_reset() {
        let codec = Codec12Hz::new_mock();
        let mut streaming = StreamingDecoder::new(codec, 5.0);

        streaming.process(&[100, 101, 102]).unwrap();
        assert!(streaming.total_samples() > 0);

        streaming.reset();
        assert_eq!(streaming.total_samples(), 0);
        assert_eq!(streaming.chunk_count(), 0);
    }

    #[test]
    fn test_streaming_decoder_continuity() {
        let codec = Codec12Hz::new_mock();
        let mut streaming = StreamingDecoder::new(codec, 5.0);

        // Process multiple chunks
        let mut all_samples = Vec::new();
        for i in 0..5 {
            let tokens: Vec<u32> = (i * 10..(i + 1) * 10).collect();
            let chunk = streaming.process(&tokens).unwrap();
            all_samples.extend_from_slice(&chunk.pcm);
        }

        // Flush remaining
        if let Ok(flushed) = streaming.flush() {
            all_samples.extend_from_slice(&flushed.pcm);
        }

        // All samples should be in valid range
        for &s in &all_samples {
            assert!((-1.0..=1.0).contains(&s), "Sample out of range: {}", s);
        }
    }

    #[test]
    fn test_into_streaming() {
        let codec = Codec12Hz::new_mock();
        let streaming = codec.into_streaming(10.0);

        assert_eq!(streaming.sample_rate(), DEFAULT_SAMPLE_RATE);
    }
}
