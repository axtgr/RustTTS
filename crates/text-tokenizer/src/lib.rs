//! # text-tokenizer
//!
//! Text tokenization for the Qwen3-TTS Rust Engine.
//!
//! This crate provides BPE/Unigram tokenization compatible with the
//! Qwen3-TTS model, including:
//! - Compatible vocabulary and merge rules
//! - Streaming encode support
//! - Offset tracking for debugging
//! - Qwen3-TTS specific special tokens
//!
//! # Example
//!
//! ```ignore
//! use text_tokenizer::Tokenizer;
//! use tts_core::{TextTokenizer, NormText, Lang};
//!
//! // Load from HuggingFace pretrained directory
//! let tokenizer = Tokenizer::from_pretrained("./models/qwen3-tts-0.6b")?;
//! let text = NormText::new("Hello world", Lang::En);
//! let tokens = tokenizer.encode(&text)?;
//! println!("Token IDs: {:?}", tokens.ids);
//! ```

use std::path::Path;

use tracing::{info, instrument};
use tts_core::{NormText, TextTokenizer, TokenSeq, TtsError, TtsResult};

/// Qwen3-TTS specific special token IDs (from config.json).
#[derive(Debug, Clone, Copy)]
pub struct Qwen3TTSTokens {
    /// TTS begin-of-sequence token ID (151672).
    pub tts_bos_token_id: u32,
    /// TTS end-of-sequence token ID (151673).
    pub tts_eos_token_id: u32,
    /// TTS padding token ID (151671).
    pub tts_pad_token_id: u32,
    /// Codec begin-of-sequence token ID (2149).
    pub codec_bos_id: u32,
    /// Codec end-of-sequence token ID (2150).
    pub codec_eos_id: u32,
    /// Codec padding token ID (2148).
    pub codec_pad_id: u32,
}

impl Default for Qwen3TTSTokens {
    fn default() -> Self {
        Self {
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            codec_bos_id: 2149,
            codec_eos_id: 2150,
            codec_pad_id: 2148,
        }
    }
}

/// Tokenizer wrapper for Qwen3-TTS compatibility.
///
/// Supports loading from:
/// - tokenizer.json (HuggingFace format)
/// - vocab.json + merges.txt (Qwen format)
/// - Pretrained model directory
#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    audio_bos_token_id: Option<u32>,
    audio_eos_token_id: Option<u32>,
    /// Qwen3-TTS specific tokens.
    qwen3_tokens: Qwen3TTSTokens,
}

impl Tokenizer {
    /// Load a tokenizer from a pretrained model directory.
    ///
    /// Looks for files in order:
    /// 1. `tokenizer.json` (full HuggingFace tokenizer)
    /// 2. `vocab.json` + `merges.txt` (BPE components)
    ///
    /// Also reads `tokenizer_config.json` for special token mappings.
    #[instrument(skip_all, fields(path = %path.as_ref().display()))]
    pub fn from_pretrained(path: impl AsRef<Path>) -> TtsResult<Self> {
        let dir = path.as_ref();

        // Try tokenizer.json first (complete tokenizer)
        let tokenizer_json = dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            info!("Loading tokenizer from tokenizer.json");
            return Self::from_file(&tokenizer_json);
        }

        // Otherwise, try vocab.json + merges.txt
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");

        if vocab_path.exists() && merges_path.exists() {
            info!("Loading tokenizer from vocab.json + merges.txt");
            return Self::from_vocab_and_merges(&vocab_path, &merges_path);
        }

        Err(TtsError::ModelLoad {
            path: dir.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no tokenizer.json or vocab.json found in directory",
            ),
        })
    }

    /// Load a tokenizer from vocab.json and merges.txt files.
    ///
    /// This is the format used by Qwen3-TTS models on HuggingFace.
    #[instrument(skip_all, fields(vocab = %vocab_path.as_ref().display()))]
    pub fn from_vocab_and_merges(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> TtsResult<Self> {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;

        let vocab_path = vocab_path.as_ref();
        let merges_path = merges_path.as_ref();

        // Build BPE model from vocab and merges
        let bpe = BPE::from_file(
            vocab_path
                .to_str()
                .ok_or_else(|| TtsError::config("invalid vocab path"))?,
            merges_path
                .to_str()
                .ok_or_else(|| TtsError::config("invalid merges path"))?,
        )
        .build()
        .map_err(|e| TtsError::ModelLoad {
            path: vocab_path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        })?;

        // Create tokenizer with BPE model
        let mut inner = tokenizers::Tokenizer::new(bpe);

        // Add byte-level pre-tokenizer (same as GPT-2/Qwen)
        inner.with_pre_tokenizer(Some(ByteLevel::default()));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        inner.with_post_processor(Some(ByteLevelProcessor::default()));

        info!(
            vocab_size = inner.get_vocab_size(true),
            "Tokenizer loaded from vocab.json + merges.txt"
        );

        Self::from_inner(inner)
    }

    /// Load a tokenizer from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> TtsResult<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| TtsError::ModelLoad {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        })?;

        Self::from_inner(inner)
    }

    /// Create a tokenizer from JSON string.
    pub fn from_json(json: &str) -> TtsResult<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| TtsError::config(format!("invalid tokenizer JSON: {e}")))?;

        Self::from_inner(inner)
    }

    /// Create tokenizer from inner tokenizers::Tokenizer.
    fn from_inner(inner: tokenizers::Tokenizer) -> TtsResult<Self> {
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

        // Qwen3-TTS specific audio tokens
        let audio_bos_token_id = inner
            .token_to_id("<|audio_bos|>")
            .or_else(|| inner.token_to_id("<|AUDIO|>"));

        let audio_eos_token_id = inner
            .token_to_id("<|audio_eos|>")
            .or_else(|| inner.token_to_id("<|/AUDIO|>"));

        // Use default Qwen3-TTS token IDs
        let qwen3_tokens = Qwen3TTSTokens::default();

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            audio_bos_token_id,
            audio_eos_token_id,
            qwen3_tokens,
        })
    }

    /// Get the underlying tokenizers::Tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// Get the audio BOS token ID (start of audio generation).
    pub fn audio_bos_token_id(&self) -> Option<u32> {
        self.audio_bos_token_id
    }

    /// Get the audio EOS token ID (end of audio generation).
    pub fn audio_eos_token_id(&self) -> Option<u32> {
        self.audio_eos_token_id
    }

    /// Get Qwen3-TTS specific token IDs.
    pub fn qwen3_tokens(&self) -> &Qwen3TTSTokens {
        &self.qwen3_tokens
    }

    /// Get the TTS begin-of-sequence token ID.
    pub fn tts_bos_token_id(&self) -> u32 {
        self.qwen3_tokens.tts_bos_token_id
    }

    /// Get the TTS end-of-sequence token ID.
    pub fn tts_eos_token_id(&self) -> u32 {
        self.qwen3_tokens.tts_eos_token_id
    }

    /// Get the codec begin-of-sequence token ID.
    pub fn codec_bos_id(&self) -> u32 {
        self.qwen3_tokens.codec_bos_id
    }

    /// Get the codec end-of-sequence token ID.
    pub fn codec_eos_id(&self) -> u32 {
        self.qwen3_tokens.codec_eos_id
    }

    /// Lookup a token ID by its string representation.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Lookup a token string by its ID.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Encode text and prepend BOS token if present.
    pub fn encode_with_bos(&self, text: &NormText) -> TtsResult<TokenSeq> {
        let mut tokens = self.encode_raw(&text.text)?;

        if let Some(bos_id) = self.bos_token_id {
            tokens.ids.insert(0, bos_id);
            tokens.offsets.insert(0, (0, 0));
        }

        Ok(tokens)
    }

    /// Encode text and append EOS token if present.
    pub fn encode_with_eos(&self, text: &NormText) -> TtsResult<TokenSeq> {
        let mut tokens = self.encode_raw(&text.text)?;

        if let Some(eos_id) = self.eos_token_id {
            let end_offset = text.text.len();
            tokens.ids.push(eos_id);
            tokens.offsets.push((end_offset, end_offset));
        }

        Ok(tokens)
    }

    /// Encode text with both BOS and EOS tokens.
    pub fn encode_with_special_tokens(&self, text: &NormText) -> TtsResult<TokenSeq> {
        let mut tokens = self.encode_raw(&text.text)?;

        if let Some(bos_id) = self.bos_token_id {
            tokens.ids.insert(0, bos_id);
            tokens.offsets.insert(0, (0, 0));
        }

        if let Some(eos_id) = self.eos_token_id {
            let end_offset = text.text.len();
            tokens.ids.push(eos_id);
            tokens.offsets.push((end_offset, end_offset));
        }

        Ok(tokens)
    }

    /// Encode raw text without the NormText wrapper.
    fn encode_raw(&self, text: &str) -> TtsResult<TokenSeq> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::tokenization(e.to_string()))?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

        Ok(TokenSeq::new(ids, offsets))
    }

    /// Encode text by sentences for streaming (low latency).
    /// Returns an iterator over token sequences for each sentence.
    pub fn encode_streaming<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = TtsResult<TokenSeq>> + 'a {
        // Split by sentence-ending punctuation
        let sentences = split_sentences(text);

        sentences.into_iter().map(move |sentence| {
            let encoding = self
                .inner
                .encode(sentence, false)
                .map_err(|e| TtsError::tokenization(e.to_string()))?;

            let ids: Vec<u32> = encoding.get_ids().to_vec();
            let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

            Ok(TokenSeq::new(ids, offsets))
        })
    }
}

/// Split text into sentences for streaming tokenization.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;

    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' || c == ';' {
            // Include the punctuation in the sentence
            let end = i + c.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end;
        }
    }

    // Add remaining text
    let remaining = text[start..].trim();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    // If no sentences found, return the whole text
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim());
    }

    sentences
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

    #[test]
    fn test_qwen3_tts_tokens_default() {
        let tokens = Qwen3TTSTokens::default();

        assert_eq!(tokens.tts_bos_token_id, 151672);
        assert_eq!(tokens.tts_eos_token_id, 151673);
        assert_eq!(tokens.tts_pad_token_id, 151671);
        assert_eq!(tokens.codec_bos_id, 2149);
        assert_eq!(tokens.codec_eos_id, 2150);
        assert_eq!(tokens.codec_pad_id, 2148);
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. How are you? I am fine!";
        let sentences = split_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
    }

    #[test]
    fn test_split_sentences_no_punctuation() {
        let text = "Hello world without punctuation";
        let sentences = split_sentences(text);

        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Hello world without punctuation");
    }

    #[test]
    fn test_split_sentences_russian() {
        let text = "Привет мир. Как дела? Всё хорошо!";
        let sentences = split_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Привет мир.");
        assert_eq!(sentences[1], "Как дела?");
        assert_eq!(sentences[2], "Всё хорошо!");
    }
}
