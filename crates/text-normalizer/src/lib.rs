//! # text-normalizer
//!
//! Text normalization pipeline for the Qwen3-TTS Rust Engine.
//!
//! This crate provides rules-based text normalization for Russian and English,
//! handling:
//! - Numbers (cardinal, ordinal)
//! - Dates and times
//! - Currency and units
//! - Abbreviations
//! - Symbol normalization
//!
//! # Example
//!
//! ```ignore
//! use text_normalizer::Normalizer;
//! use tts_core::{TextNormalizer, Lang};
//!
//! let normalizer = Normalizer::new();
//! let result = normalizer.normalize("100 рублей", Some(Lang::Ru))?;
//! assert_eq!(result.text, "сто рублей");
//! ```

mod rules;

use tracing::instrument;
use tts_core::{Lang, NormText, TextNormalizer, TtsError, TtsResult};

pub use rules::Rule;

/// Text normalizer with configurable rule pipeline.
#[derive(Debug)]
pub struct Normalizer {
    rules: Vec<Box<dyn Rule>>,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Normalizer {
    /// Create a new normalizer with default rules.
    pub fn new() -> Self {
        Self {
            rules: rules::default_rules(),
        }
    }

    /// Create a normalizer with custom rules.
    pub fn with_rules(rules: Vec<Box<dyn Rule>>) -> Self {
        Self { rules }
    }

    /// Add a rule to the pipeline.
    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        self.rules.push(rule);
    }

    /// Detect the language of the input text.
    fn detect_language(&self, input: &str) -> Lang {
        let cyrillic_count = input.chars().filter(|c| is_cyrillic(*c)).count();
        let latin_count = input.chars().filter(|c| is_latin(*c)).count();

        match cyrillic_count.cmp(&latin_count) {
            std::cmp::Ordering::Greater => Lang::Ru,
            std::cmp::Ordering::Less => Lang::En,
            std::cmp::Ordering::Equal => {
                // When equal, default to Russian if any letters present, else Ru
                if cyrillic_count > 0 {
                    Lang::Mixed
                } else {
                    Lang::Ru // No alphabetic chars, default to Russian
                }
            }
        }
    }
}

impl TextNormalizer for Normalizer {
    #[instrument(skip(self), fields(input_len = input.len()))]
    fn normalize(&self, input: &str, lang_hint: Option<Lang>) -> TtsResult<NormText> {
        if input.is_empty() {
            return Err(TtsError::invalid_input("empty input text"));
        }

        let lang = lang_hint.unwrap_or_else(|| self.detect_language(input));
        let mut text = input.to_string();

        // Apply all rules in sequence
        for rule in &self.rules {
            if rule.applies_to(lang) {
                text = rule.apply(&text, lang)?;
            }
        }

        Ok(NormText::new(text, lang))
    }
}

/// Check if a character is Cyrillic.
fn is_cyrillic(c: char) -> bool {
    matches!(c, '\u{0400}'..='\u{04FF}')
}

/// Check if a character is Latin.
fn is_latin(c: char) -> bool {
    c.is_ascii_alphabetic()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_creation() {
        let normalizer = Normalizer::new();
        assert!(!normalizer.rules.is_empty());
    }

    #[test]
    fn test_language_detection() {
        let normalizer = Normalizer::new();

        assert_eq!(normalizer.detect_language("Привет мир"), Lang::Ru);
        assert_eq!(normalizer.detect_language("Hello world"), Lang::En);
        // "Привет" = 6 cyrillic, "world" = 5 latin, so cyrillic wins -> Ru
        assert_eq!(normalizer.detect_language("Привет world"), Lang::Ru);
        // Equal counts -> Mixed
        assert_eq!(normalizer.detect_language("Привет worlds"), Lang::Mixed);
    }

    #[test]
    fn test_empty_input_error() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_basic_normalization() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("Hello", Some(Lang::En)).unwrap();
        assert_eq!(result.lang, Lang::En);
        assert!(!result.text.is_empty());
    }
}
