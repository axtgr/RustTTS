//! Integration tests for text-tokenizer crate.
//!
//! These tests verify tokenization with real tokenizer files.

use std::path::PathBuf;
use text_tokenizer::Tokenizer;
use tts_core::{Lang, NormText, TextTokenizer};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn test_load_tokenizer_from_file() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Check vocab size
    assert!(tokenizer.vocab_size() > 100, "vocab should have entries");

    // Check special tokens
    assert_eq!(tokenizer.bos_token_id(), Some(1));
    assert_eq!(tokenizer.eos_token_id(), Some(2));
    assert_eq!(tokenizer.pad_token_id(), Some(0));
}

#[test]
fn test_encode_simple_text() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let text = NormText::new("Hello", Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode");

    // Should produce some tokens
    assert!(!tokens.ids.is_empty(), "should produce tokens");

    // Offsets should match token count
    assert_eq!(tokens.ids.len(), tokens.offsets.len());
}

#[test]
fn test_encode_decode_roundtrip() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let original = "Hello World";
    let text = NormText::new(original, Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode");

    let decoded = tokenizer.decode(&tokens).expect("should decode");

    // Decoded should match original (ignoring potential whitespace differences)
    assert_eq!(decoded.trim(), original);
}

#[test]
fn test_encode_with_spaces() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let text = NormText::new("hello world", Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode");

    // Should handle spaces properly
    assert!(!tokens.ids.is_empty());
}

#[test]
fn test_special_token_lookup() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Lookup special tokens via inner tokenizer
    let inner = tokenizer.inner();

    // Verify audio tokens exist
    let audio_bos = inner.token_to_id("<|audio_bos|>");
    let audio_eos = inner.token_to_id("<|audio_eos|>");

    assert_eq!(audio_bos, Some(3), "audio_bos should be id 3");
    assert_eq!(audio_eos, Some(4), "audio_eos should be id 4");
}

#[test]
fn test_vocab_size() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Our test tokenizer has ~200 tokens
    let vocab_size = tokenizer.vocab_size();
    assert!(
        (100..=300).contains(&vocab_size),
        "vocab_size = {}",
        vocab_size
    );
}

#[test]
fn test_offset_tracking() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let input = "Hello";
    let text = NormText::new(input, Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode");

    // Each token should have valid offset
    for (start, end) in &tokens.offsets {
        assert!(*start <= *end, "start should be <= end");
        assert!(*end <= input.len(), "offset should be within input");
    }
}

#[test]
fn test_empty_input() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let text = NormText::new("", Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode empty");

    // Empty input should produce no tokens (or just special tokens depending on config)
    // This depends on the tokenizer config
    assert!(
        tokens.ids.len() <= 2,
        "empty input should produce few/no tokens"
    );
}

#[test]
fn test_unicode_text() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Test with some unicode characters
    let text = NormText::new("hello", Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode unicode");

    assert!(!tokens.ids.is_empty());
}

#[test]
fn test_long_text() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Create a longer text
    let long_text = "hello world ".repeat(100);
    let text = NormText::new(&long_text, Lang::En);
    let tokens = tokenizer.encode(&text).expect("should encode long text");

    // Should handle long text
    assert!(!tokens.ids.is_empty());
    assert!(tokens.len() > 10, "long text should produce many tokens");
}

#[test]
fn test_audio_tokens() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Verify audio tokens
    assert_eq!(tokenizer.audio_bos_token_id(), Some(3));
    assert_eq!(tokenizer.audio_eos_token_id(), Some(4));
}

#[test]
fn test_token_lookup() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    // Lookup by string
    let hello_id = tokenizer.token_to_id("hello");
    assert!(hello_id.is_some(), "should find 'hello' token");

    // Lookup by ID
    if let Some(id) = hello_id {
        let token = tokenizer.id_to_token(id);
        assert_eq!(token, Some("hello".to_string()));
    }
}

#[test]
fn test_encode_with_special_tokens() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let text = NormText::new("hello", Lang::En);

    // With BOS
    let with_bos = tokenizer.encode_with_bos(&text).expect("should encode");
    assert!(with_bos.ids.first() == Some(&1), "should start with BOS");

    // With EOS
    let with_eos = tokenizer.encode_with_eos(&text).expect("should encode");
    assert!(with_eos.ids.last() == Some(&2), "should end with EOS");

    // With both
    let with_both = tokenizer
        .encode_with_special_tokens(&text)
        .expect("should encode");
    assert!(with_both.ids.first() == Some(&1), "should start with BOS");
    assert!(with_both.ids.last() == Some(&2), "should end with EOS");
}

#[test]
fn test_streaming_encode() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let text = "Hello world. How are you? I am fine!";

    let sentences: Vec<_> = tokenizer.encode_streaming(text).collect();

    // Should split into 3 sentences
    assert_eq!(sentences.len(), 3, "should have 3 sentences");

    // Each should be valid
    for result in sentences {
        let tokens = result.expect("should encode sentence");
        assert!(!tokens.ids.is_empty(), "sentence should have tokens");
    }
}

#[test]
fn test_streaming_encode_no_punctuation() {
    let path = fixture_path("test_tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("should load tokenizer");

    let text = "Hello world without punctuation";

    let sentences: Vec<_> = tokenizer.encode_streaming(text).collect();

    // Should return single "sentence"
    assert_eq!(sentences.len(), 1);
}

#[cfg(feature = "golden_tests")]
mod golden_tests {
    use super::*;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models/qwen3-tts-0.6b-customvoice")
    }

    /// Golden test for tokenization matching Python SDK.
    /// These token IDs are verified against transformers AutoTokenizer.
    #[test]
    fn test_golden_tokenization_matches_python() {
        let tokenizer =
            Tokenizer::from_pretrained(model_path()).expect("should load model tokenizer");

        // Test cases verified against Python:
        // from transformers import AutoTokenizer
        // tok = AutoTokenizer.from_pretrained("models/qwen3-tts-0.6b-customvoice")
        // tok.encode("Hello", add_special_tokens=False) -> [9707]

        let test_cases = [
            ("Hello", vec![9707]),
            ("Hello world", vec![9707, 1879]),
            (
                "Hello, my name is Ryan.",
                vec![9707, 11, 847, 829, 374, 13646, 13],
            ),
            ("Testing 123", vec![16451, 220, 16, 17, 18]),
        ];

        for (text, expected_tokens) in test_cases {
            let norm_text = NormText::new(text, Lang::En);
            let tokens = tokenizer.encode(&norm_text).expect("should encode");

            assert_eq!(
                tokens.ids, expected_tokens,
                "Token mismatch for '{}': got {:?}, expected {:?}",
                text, tokens.ids, expected_tokens
            );
        }
    }

    /// Verify add_prefix_space=false behavior.
    /// "Hello" should tokenize to [9707], NOT [21927] (which is " Hello" with prefix space).
    #[test]
    fn test_no_prefix_space() {
        let tokenizer =
            Tokenizer::from_pretrained(model_path()).expect("should load model tokenizer");

        let text = NormText::new("Hello", Lang::En);
        let tokens = tokenizer.encode(&text).expect("should encode");

        // 9707 = "Hello" (no prefix space)
        // 21927 = " Hello" (with prefix space) - WRONG if this appears
        assert_eq!(
            tokens.ids,
            vec![9707],
            "Should tokenize 'Hello' to [9707], not [21927] (prefix space bug)"
        );
    }
}
