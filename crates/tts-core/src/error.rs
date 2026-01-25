//! Unified error types for the TTS engine.

use std::path::PathBuf;

/// Main error type for TTS operations.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// Text normalization failed.
    #[error("normalization failed: {0}")]
    Normalization(String),

    /// Tokenization failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// Model loading error.
    #[error("model load failed for {path}: {source}")]
    ModelLoad {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Model inference error.
    #[error("inference error: {0}")]
    Inference(String),

    /// Audio decoding error.
    #[error("audio decode error: {0}")]
    AudioDecode(String),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Timeout during operation.
    #[error("operation timeout after {ms}ms")]
    Timeout { ms: u64 },

    /// Resource exhausted (e.g., KV cache full).
    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Invalid input provided.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Internal error (should not happen in normal operation).
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience type alias for Results with TtsError.
pub type TtsResult<T> = Result<T, TtsError>;

impl TtsError {
    /// Create a normalization error with message.
    pub fn normalization(msg: impl Into<String>) -> Self {
        Self::Normalization(msg.into())
    }

    /// Create a tokenization error with message.
    pub fn tokenization(msg: impl Into<String>) -> Self {
        Self::Tokenization(msg.into())
    }

    /// Create an inference error with message.
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create an audio decode error with message.
    pub fn audio_decode(msg: impl Into<String>) -> Self {
        Self::AudioDecode(msg.into())
    }

    /// Create a config error with message.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create an invalid input error with message.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create an internal error with message.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TtsError::normalization("invalid character");
        assert_eq!(err.to_string(), "normalization failed: invalid character");

        let err = TtsError::Timeout { ms: 5000 };
        assert_eq!(err.to_string(), "operation timeout after 5000ms");
    }

    #[test]
    fn test_error_constructors() {
        let err = TtsError::tokenization("unknown token");
        assert!(matches!(err, TtsError::Tokenization(_)));

        let err = TtsError::inference("model failed");
        assert!(matches!(err, TtsError::Inference(_)));
    }
}
