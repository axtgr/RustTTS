//! # text-tokenizer
//!
//! Text tokenization for the Qwen3-TTS Rust Engine.
//!
//! This crate provides BPE/Unigram tokenization compatible with the
//! Qwen3-TTS model, including:
//! - Compatible vocabulary and merge rules
//! - Streaming encode support
//! - Offset tracking for debugging
//!
//! # Example
//!
//! ```ignore
//! use text_tokenizer::Tokenizer;
//! use tts_core::{TextTokenizer, NormText, Lang};
//!
//! let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//! let text = NormText::new("Hello world", Lang::En);
//! let tokens = tokenizer.encode(&text)?;
//! println!("Token IDs: {:?}", tokens.ids);
//! ```

use std::path::Path;

use tracing::instrument;
use tts_core::{NormText, TextTokenizer, TokenSeq, TtsError, TtsResult};

/// Tokenizer wrapper for Qwen3-TTS compatibility.
#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> TtsResult<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| TtsError::ModelLoad {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        })?;

        let bos_token_id = inner
            .token_to_id("<|startoftext|>")
            .or_else(|| inner.token_to_id("<s>"))
            .or_else(|| inner.token_to_id("<bos>"));

        let eos_token_id = inner
            .token_to_id("<|endoftext|>")
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("<eos>"));

        let pad_token_id = inner
            .token_to_id("<pad>")
            .or_else(|| inner.token_to_id("<|pad|>"));

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Create a tokenizer from JSON string.
    pub fn from_json(json: &str) -> TtsResult<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| TtsError::config(format!("invalid tokenizer JSON: {e}")))?;

        let bos_token_id = inner
            .token_to_id("<|startoftext|>")
            .or_else(|| inner.token_to_id("<s>"))
            .or_else(|| inner.token_to_id("<bos>"));

        let eos_token_id = inner
            .token_to_id("<|endoftext|>")
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("<eos>"));

        let pad_token_id = inner
            .token_to_id("<pad>")
            .or_else(|| inner.token_to_id("<|pad|>"));

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Get the underlying tokenizers::Tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

impl TextTokenizer for Tokenizer {
    #[instrument(skip(self, text), fields(text_len = text.text.len()))]
    fn encode(&self, text: &NormText) -> TtsResult<TokenSeq> {
        let encoding = self
            .inner
            .encode(text.text.as_str(), true)
            .map_err(|e| TtsError::tokenization(e.to_string()))?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

        Ok(TokenSeq::new(ids, offsets))
    }

    #[instrument(skip(self, tokens), fields(num_tokens = tokens.len()))]
    fn decode(&self, tokens: &TokenSeq) -> TtsResult<String> {
        self.inner
            .decode(&tokens.ids, true)
            .map_err(|e| TtsError::tokenization(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

/// A mock tokenizer for testing without model files.
#[derive(Debug, Default)]
pub struct MockTokenizer {
    vocab_size: usize,
}

impl MockTokenizer {
    /// Create a new mock tokenizer.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl TextTokenizer for MockTokenizer {
    fn encode(&self, text: &NormText) -> TtsResult<TokenSeq> {
        // Simple character-based mock encoding
        let ids: Vec<u32> = text
            .text
            .chars()
            .map(|c| (c as u32) % self.vocab_size as u32)
            .collect();

        let offsets: Vec<(usize, usize)> = text
            .text
            .char_indices()
            .map(|(i, c)| (i, i + c.len_utf8()))
            .collect();

        Ok(TokenSeq::new(ids, offsets))
    }

    fn decode(&self, tokens: &TokenSeq) -> TtsResult<String> {
        // Mock decode - just convert IDs to characters
        let text: String = tokens
            .ids
            .iter()
            .map(|&id| char::from_u32(id + 'a' as u32).unwrap_or('?'))
            .collect();
        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(1)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(2)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_core::Lang;

    #[test]
    fn test_mock_tokenizer_encode() {
        let tokenizer = MockTokenizer::new(256);
        let text = NormText::new("hello", Lang::En);
        let tokens = tokenizer.encode(&text).unwrap();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens.offsets.len(), 5);
    }

    #[test]
    fn test_mock_tokenizer_special_tokens() {
        let tokenizer = MockTokenizer::new(256);

        assert_eq!(tokenizer.bos_token_id(), Some(1));
        assert_eq!(tokenizer.eos_token_id(), Some(2));
        assert_eq!(tokenizer.pad_token_id(), Some(0));
        assert_eq!(tokenizer.vocab_size(), 256);
    }
}
