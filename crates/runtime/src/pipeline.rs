//! TTS Pipeline - full text-to-speech processing pipeline.
//!
//! Combines all components: normalizer → tokenizer → acoustic model → codec.

use tracing::{debug, info, instrument};

use audio_codec_12hz::{Codec12Hz, DEFAULT_CROSSFADE_MS, StreamingDecoder};
use text_normalizer::Normalizer;
use text_tokenizer::MockTokenizer;
use tts_core::{
    AudioChunk, AudioCodec, Lang, NormText, SynthesisRequest, TextNormalizer, TextTokenizer,
    TtsError, TtsResult,
};

/// Backend selection for the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// Use mock components (for testing without model weights).
    Mock,
    /// Use real neural network components.
    Neural,
}

/// Configuration for the TTS pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Backend selection.
    pub backend: PipelineBackend,
    /// Default language for text normalization.
    pub default_lang: Lang,
    /// Crossfade duration in milliseconds for streaming.
    pub crossfade_ms: f32,
    /// Default chunk size in tokens for streaming.
    pub chunk_tokens: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            backend: PipelineBackend::Mock,
            default_lang: Lang::Ru,
            crossfade_ms: DEFAULT_CROSSFADE_MS,
            chunk_tokens: 10,
            max_seq_len: 4096,
        }
    }
}

impl PipelineConfig {
    /// Create a mock pipeline configuration.
    pub fn mock() -> Self {
        Self {
            backend: PipelineBackend::Mock,
            ..Default::default()
        }
    }

    /// Create a neural pipeline configuration.
    pub fn neural() -> Self {
        Self {
            backend: PipelineBackend::Neural,
            ..Default::default()
        }
    }
}

/// The main TTS pipeline combining all processing stages.
///
/// This is a mock implementation for Phase 4.
/// Full neural implementation will be added when model weights are available.
pub struct TtsPipeline {
    normalizer: Normalizer,
    tokenizer: MockTokenizer,
    codec: Codec12Hz,
    config: PipelineConfig,
}

impl std::fmt::Debug for TtsPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsPipeline")
            .field("config", &self.config)
            .finish()
    }
}

impl TtsPipeline {
    /// Create a new mock pipeline for testing.
    pub fn new_mock() -> TtsResult<Self> {
        let config = PipelineConfig::mock();

        info!("Creating mock TTS pipeline");

        Ok(Self {
            normalizer: Normalizer::new(),
            tokenizer: MockTokenizer::new(65536),
            codec: Codec12Hz::new_mock(),
            config,
        })
    }

    /// Create a neural pipeline with loaded models.
    ///
    /// Note: This is a placeholder. Full implementation requires model weights.
    #[instrument(skip(_acoustic_model_path, _tokenizer_path, _codec_path))]
    pub fn new_neural(
        _acoustic_model_path: impl AsRef<std::path::Path>,
        _tokenizer_path: impl AsRef<std::path::Path>,
        _codec_path: impl AsRef<std::path::Path>,
    ) -> TtsResult<Self> {
        // For now, return mock pipeline
        // Full neural implementation will be added in later phases
        info!("Neural pipeline requested, using mock for now");
        Self::new_mock()
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Normalize text for the given language.
    pub fn normalize(&self, text: &str, lang: Option<Lang>) -> TtsResult<NormText> {
        let lang = lang.unwrap_or(self.config.default_lang);
        self.normalizer.normalize(text, Some(lang))
    }

    /// Tokenize normalized text.
    pub fn tokenize(&self, text: &NormText) -> TtsResult<Vec<u32>> {
        let seq = self.tokenizer.encode(text)?;
        Ok(seq.ids)
    }

    /// Generate acoustic tokens from text tokens.
    ///
    /// Note: This is a mock implementation that generates pseudo-random tokens.
    pub fn generate_acoustic(&self, text_tokens: &[u32], max_tokens: usize) -> TtsResult<Vec<u32>> {
        if text_tokens.is_empty() {
            return Err(TtsError::invalid_input("empty text tokens"));
        }

        // Mock: generate acoustic tokens based on input
        // Real implementation would use the acoustic model
        let num_tokens = (text_tokens.len() * 2).min(max_tokens);
        let acoustic_tokens: Vec<u32> = text_tokens
            .iter()
            .cycle()
            .take(num_tokens)
            .map(|&t| (t * 7 + 13) % 65536)
            .collect();

        debug!(
            input_tokens = text_tokens.len(),
            output_tokens = acoustic_tokens.len(),
            "Generated acoustic tokens (mock)"
        );

        Ok(acoustic_tokens)
    }

    /// Decode acoustic tokens to audio.
    pub fn decode_audio(&self, tokens: &[u32]) -> TtsResult<AudioChunk> {
        self.codec.decode(tokens)
    }

    /// Full synthesis: text → audio.
    #[instrument(skip(self), fields(text_len = text.len()))]
    pub fn synthesize(&self, text: &str, lang: Option<Lang>) -> TtsResult<AudioChunk> {
        // 1. Normalize
        let normalized = self.normalize(text, lang)?;
        debug!(normalized = %normalized.text, "Text normalized");

        // 2. Tokenize
        let text_tokens = self.tokenize(&normalized)?;
        debug!(num_tokens = text_tokens.len(), "Text tokenized");

        // 3. Generate acoustic tokens
        let acoustic_tokens = self.generate_acoustic(&text_tokens, self.config.max_seq_len)?;
        debug!(
            num_acoustic = acoustic_tokens.len(),
            "Acoustic tokens generated"
        );

        // 4. Decode to audio
        let audio = self.decode_audio(&acoustic_tokens)?;
        debug!(
            samples = audio.num_samples(),
            duration_ms = audio.duration_ms(),
            "Audio decoded"
        );

        Ok(audio)
    }

    /// Synthesize from a SynthesisRequest.
    pub fn synthesize_request(&self, request: &SynthesisRequest) -> TtsResult<AudioChunk> {
        self.synthesize(&request.text, Some(request.lang))
    }

    /// Create a streaming session for incremental synthesis.
    pub fn streaming_session(&self) -> TtsResult<StreamingSession<'_>> {
        StreamingSession::new(self)
    }
}

/// A streaming synthesis session for incremental audio generation.
pub struct StreamingSession<'a> {
    pipeline: &'a TtsPipeline,
    decoder: StreamingDecoder,
    text_tokens: Vec<u32>,
    generated_acoustic: Vec<u32>,
    position: usize,
    finished: bool,
}

impl<'a> StreamingSession<'a> {
    /// Create a new streaming session.
    fn new(pipeline: &'a TtsPipeline) -> TtsResult<Self> {
        let decoder = StreamingDecoder::new(Codec12Hz::new_mock(), pipeline.config.crossfade_ms);

        Ok(Self {
            pipeline,
            decoder,
            text_tokens: Vec::new(),
            generated_acoustic: Vec::new(),
            position: 0,
            finished: false,
        })
    }

    /// Set the text to synthesize.
    #[instrument(skip(self), fields(text_len = text.len()))]
    pub fn set_text(&mut self, text: &str, lang: Option<Lang>) -> TtsResult<()> {
        // Normalize and tokenize
        let normalized = self.pipeline.normalize(text, lang)?;
        self.text_tokens = self.pipeline.tokenize(&normalized)?;

        // Pre-generate all acoustic tokens for streaming
        self.generated_acoustic = self
            .pipeline
            .generate_acoustic(&self.text_tokens, self.pipeline.config.max_seq_len)?;

        debug!(
            text_tokens = self.text_tokens.len(),
            acoustic_tokens = self.generated_acoustic.len(),
            "Streaming session initialized"
        );

        self.position = 0;
        self.finished = false;

        Ok(())
    }

    /// Generate the next audio chunk.
    pub fn next_chunk(&mut self) -> TtsResult<Option<AudioChunk>> {
        if self.finished {
            return Ok(None);
        }

        if self.generated_acoustic.is_empty() {
            return Err(TtsError::invalid_input("call set_text() first"));
        }

        let chunk_tokens = self.pipeline.config.chunk_tokens;
        let end_pos = (self.position + chunk_tokens).min(self.generated_acoustic.len());

        if self.position >= end_pos {
            // No more tokens, flush the decoder
            self.finished = true;
            return match self.decoder.flush() {
                Ok(chunk) => Ok(Some(chunk)),
                Err(_) => Ok(None),
            };
        }

        let acoustic_chunk = &self.generated_acoustic[self.position..end_pos];
        self.position = end_pos;

        // Decode to audio with crossfade
        let audio = self.decoder.process(acoustic_chunk)?;

        debug!(
            position = self.position,
            total = self.generated_acoustic.len(),
            samples = audio.num_samples(),
            "Generated streaming chunk"
        );

        // Check if we've processed all tokens
        if self.position >= self.generated_acoustic.len() {
            self.finished = true;
        }

        Ok(Some(audio))
    }

    /// Check if all tokens have been processed.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get total acoustic tokens generated.
    pub fn total_acoustic_tokens(&self) -> usize {
        self.generated_acoustic.len()
    }

    /// Get current position in token stream.
    pub fn current_position(&self) -> usize {
        self.position
    }

    /// Get total audio samples produced so far.
    pub fn total_samples(&self) -> usize {
        self.decoder.total_samples()
    }

    /// Reset the session for reuse.
    pub fn reset(&mut self) {
        self.decoder.reset();
        self.text_tokens.clear();
        self.generated_acoustic.clear();
        self.position = 0;
        self.finished = false;
    }
}

impl std::fmt::Debug for StreamingSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingSession")
            .field("text_tokens", &self.text_tokens.len())
            .field("generated_acoustic", &self.generated_acoustic.len())
            .field("position", &self.position)
            .field("finished", &self.finished)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.backend, PipelineBackend::Mock);
        assert_eq!(config.default_lang, Lang::Ru);
    }

    #[test]
    fn test_pipeline_mock_creation() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        assert_eq!(pipeline.config().backend, PipelineBackend::Mock);
    }

    #[test]
    fn test_pipeline_normalize() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let normalized = pipeline.normalize("Привет мир", Some(Lang::Ru)).unwrap();
        assert!(!normalized.text.is_empty());
    }

    #[test]
    fn test_pipeline_tokenize() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let normalized = pipeline.normalize("Hello world", Some(Lang::En)).unwrap();
        let tokens = pipeline.tokenize(&normalized).unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_pipeline_generate_acoustic() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let text_tokens = vec![1, 2, 3, 4, 5];
        let acoustic = pipeline.generate_acoustic(&text_tokens, 100).unwrap();

        assert!(!acoustic.is_empty());
        assert!(acoustic.len() <= 100);
    }

    #[test]
    fn test_pipeline_full_synthesis() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let audio = pipeline.synthesize("Тест синтеза", Some(Lang::Ru)).unwrap();

        assert!(audio.num_samples() > 0);
        assert_eq!(audio.sample_rate, 24000);
    }

    #[test]
    fn test_pipeline_synthesis_request() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let request = SynthesisRequest::new("Test synthesis").with_lang(Lang::En);

        let audio = pipeline.synthesize_request(&request).unwrap();
        assert!(audio.num_samples() > 0);
    }

    #[test]
    fn test_streaming_session_creation() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let session = pipeline.streaming_session();
        assert!(session.is_ok());
    }

    #[test]
    fn test_streaming_session_set_text() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        let result = session.set_text("Привет", Some(Lang::Ru));
        assert!(result.is_ok());
        assert!(session.total_acoustic_tokens() > 0);
    }

    #[test]
    fn test_streaming_session_next_chunk() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        session.set_text("Привет мир", Some(Lang::Ru)).unwrap();

        // Get first chunk
        let chunk = session.next_chunk().unwrap();
        assert!(chunk.is_some());

        let audio = chunk.unwrap();
        assert!(audio.num_samples() > 0);
    }

    #[test]
    fn test_streaming_session_full_iteration() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        session.set_text("Test", Some(Lang::En)).unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk) = session.next_chunk().unwrap() {
            chunks.push(chunk);
            if session.is_finished() {
                break;
            }
        }

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_streaming_session_reset() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        session.set_text("Test", Some(Lang::En)).unwrap();
        session.next_chunk().unwrap();

        session.reset();
        assert_eq!(session.total_acoustic_tokens(), 0);
        assert_eq!(session.total_samples(), 0);
        assert!(!session.is_finished());
    }
}
