//! Core data types for the TTS pipeline.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Supported languages for TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    /// Russian language.
    #[default]
    Ru,
    /// English language.
    En,
    /// Mixed (auto-detect per segment).
    Mixed,
}

impl std::fmt::Display for Lang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lang::Ru => write!(f, "ru"),
            Lang::En => write!(f, "en"),
            Lang::Mixed => write!(f, "mixed"),
        }
    }
}

/// Span information for tracking text segments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanInfo {
    /// Start byte offset in original text.
    pub start: usize,
    /// End byte offset in original text.
    pub end: usize,
    /// Language of this span.
    pub lang: Lang,
    /// Type of content (e.g., "text", "number", "date").
    pub kind: String,
}

/// Normalized text with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormText {
    /// The normalized text content.
    pub text: String,
    /// Detected or specified language.
    pub lang: Lang,
    /// Span information for each segment.
    pub spans: Vec<SpanInfo>,
}

impl NormText {
    /// Create a new NormText with given text and language.
    pub fn new(text: impl Into<String>, lang: Lang) -> Self {
        Self {
            text: text.into(),
            lang,
            spans: Vec::new(),
        }
    }

    /// Create NormText with span information.
    pub fn with_spans(text: impl Into<String>, lang: Lang, spans: Vec<SpanInfo>) -> Self {
        Self {
            text: text.into(),
            lang,
            spans,
        }
    }
}

/// Token sequence with offset mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSeq {
    /// Token IDs.
    pub ids: Vec<u32>,
    /// Byte offsets mapping tokens to original text positions.
    pub offsets: Vec<(usize, usize)>,
}

impl TokenSeq {
    /// Create a new token sequence.
    pub fn new(ids: Vec<u32>, offsets: Vec<(usize, usize)>) -> Self {
        Self { ids, offsets }
    }

    /// Create an empty token sequence.
    pub fn empty() -> Self {
        Self {
            ids: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// Get the number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// A single acoustic model generation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticStep {
    /// Generated acoustic token.
    pub token: u32,
    /// Probability/confidence of the token.
    pub prob: f32,
    /// Timestamp in milliseconds.
    pub t_ms: f32,
}

/// A chunk of decoded audio.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// PCM samples (f32, mono).
    pub pcm: Arc<[f32]>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Start time in milliseconds.
    pub start_ms: f32,
    /// End time in milliseconds.
    pub end_ms: f32,
}

impl AudioChunk {
    /// Create a new audio chunk.
    pub fn new(pcm: Vec<f32>, sample_rate: u32, start_ms: f32, end_ms: f32) -> Self {
        Self {
            pcm: pcm.into(),
            sample_rate,
            start_ms,
            end_ms,
        }
    }

    /// Get the duration of this chunk in milliseconds.
    pub fn duration_ms(&self) -> f32 {
        self.end_ms - self.start_ms
    }

    /// Get the number of samples in this chunk.
    pub fn num_samples(&self) -> usize {
        self.pcm.len()
    }
}

/// Priority level for synthesis requests.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub enum Priority {
    /// Low priority (batch processing).
    Low,
    /// Normal priority (default).
    #[default]
    Normal,
    /// High priority (real-time streaming).
    High,
    /// Critical priority (immediate processing).
    Critical,
}

/// A synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisRequest {
    /// Unique session identifier.
    pub session_id: Uuid,
    /// Text to synthesize.
    pub text: String,
    /// Target language.
    pub lang: Lang,
    /// Speaker ID (if model supports multiple speakers).
    pub speaker_id: Option<u32>,
    /// Target chunk duration in milliseconds.
    pub chunk_ms: u32,
    /// Maximum latency for first audio chunk in milliseconds.
    pub max_latency_ms: Option<u64>,
    /// Request priority.
    pub priority: Priority,
    /// Random seed for deterministic generation.
    pub seed: Option<u64>,
}

impl SynthesisRequest {
    /// Create a new synthesis request with default settings.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            text: text.into(),
            lang: Lang::default(),
            speaker_id: None,
            chunk_ms: 50,
            max_latency_ms: None,
            priority: Priority::default(),
            seed: None,
        }
    }

    /// Set the language.
    pub fn with_lang(mut self, lang: Lang) -> Self {
        self.lang = lang;
        self
    }

    /// Set the speaker ID.
    pub fn with_speaker(mut self, speaker_id: u32) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    /// Set the chunk duration.
    pub fn with_chunk_ms(mut self, chunk_ms: u32) -> Self {
        self.chunk_ms = chunk_ms;
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the seed for deterministic generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lang_display() {
        assert_eq!(Lang::Ru.to_string(), "ru");
        assert_eq!(Lang::En.to_string(), "en");
        assert_eq!(Lang::Mixed.to_string(), "mixed");
    }

    #[test]
    fn test_norm_text_creation() {
        let text = NormText::new("Hello world", Lang::En);
        assert_eq!(text.text, "Hello world");
        assert_eq!(text.lang, Lang::En);
        assert!(text.spans.is_empty());
    }

    #[test]
    fn test_token_seq() {
        let seq = TokenSeq::new(vec![1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());

        let empty = TokenSeq::empty();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_audio_chunk() {
        let chunk = AudioChunk::new(vec![0.0; 1000], 16000, 0.0, 62.5);
        assert_eq!(chunk.duration_ms(), 62.5);
        assert_eq!(chunk.num_samples(), 1000);
    }

    #[test]
    fn test_synthesis_request_builder() {
        let req = SynthesisRequest::new("Test text")
            .with_lang(Lang::Ru)
            .with_speaker(1)
            .with_chunk_ms(100)
            .with_priority(Priority::High)
            .with_seed(42);

        assert_eq!(req.text, "Test text");
        assert_eq!(req.lang, Lang::Ru);
        assert_eq!(req.speaker_id, Some(1));
        assert_eq!(req.chunk_ms, 100);
        assert_eq!(req.priority, Priority::High);
        assert_eq!(req.seed, Some(42));
    }
}
