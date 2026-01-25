//! Normalization rules.

use tts_core::{Lang, TtsResult};

/// A text normalization rule.
pub trait Rule: Send + Sync + std::fmt::Debug {
    /// Get the rule name.
    fn name(&self) -> &str;

    /// Check if this rule applies to the given language.
    fn applies_to(&self, lang: Lang) -> bool;

    /// Apply the rule to the input text.
    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String>;
}

/// Create the default set of normalization rules.
pub fn default_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(WhitespaceRule),
        Box::new(UnicodeNormalizationRule),
        Box::new(SymbolRule),
        // TODO: Add more rules in Phase 1
        // - NumberRule
        // - DateRule
        // - CurrencyRule
        // - AbbreviationRule
    ]
}

/// Normalize whitespace (collapse multiple spaces, trim).
#[derive(Debug)]
pub struct WhitespaceRule;

impl Rule for WhitespaceRule {
    fn name(&self) -> &str {
        "whitespace"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, _lang: Lang) -> TtsResult<String> {
        // Collapse multiple whitespace into single spaces and trim
        let result: String = input.split_whitespace().collect::<Vec<_>>().join(" ");
        Ok(result)
    }
}

/// Unicode normalization (NFC form).
#[derive(Debug)]
pub struct UnicodeNormalizationRule;

impl Rule for UnicodeNormalizationRule {
    fn name(&self) -> &str {
        "unicode_normalization"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, _lang: Lang) -> TtsResult<String> {
        // Basic Unicode cleanup
        // Note: Full NFC normalization would require the `unicode-normalization` crate
        // For now, just handle common cases
        let result = input
            .replace('\u{00A0}', " ") // Non-breaking space -> regular space
            .replace(['\u{2019}', '\u{2018}'], "'") // Left single quote -> apostrophe
            .replace(['\u{201C}', '\u{201D}'], "\"") // Right double quote -> straight quote
            .replace('\u{2014}', " - ") // Em dash -> hyphen with spaces
            .replace('\u{2013}', "-") // En dash -> hyphen
            .replace('\u{2026}', "..."); // Ellipsis -> three dots

        Ok(result)
    }
}

/// Symbol normalization (replace or remove special symbols).
#[derive(Debug)]
pub struct SymbolRule;

impl Rule for SymbolRule {
    fn name(&self) -> &str {
        "symbol"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, _lang: Lang) -> TtsResult<String> {
        let mut result = String::with_capacity(input.len());

        for c in input.chars() {
            match c {
                // Keep alphanumeric, whitespace, and basic punctuation
                _ if c.is_alphanumeric() => result.push(c),
                ' ' | '\t' | '\n' | '\r' => result.push(c),
                '.' | ',' | '!' | '?' | ':' | ';' | '-' | '\'' | '"' | '(' | ')' => result.push(c),
                // Replace common symbols
                '@' => result.push_str(" at "),
                '&' => result.push_str(" and "),
                '%' => result.push_str(" percent "),
                '+' => result.push_str(" plus "),
                '=' => result.push_str(" equals "),
                '#' => result.push_str(" number "),
                // Skip other symbols
                _ => {}
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_rule() {
        let rule = WhitespaceRule;
        let result = rule.apply("  hello   world  ", Lang::En).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_unicode_normalization_rule() {
        let rule = UnicodeNormalizationRule;

        // Test em dash
        let result = rule.apply("hello—world", Lang::En).unwrap();
        assert_eq!(result, "hello - world");

        // Test smart quotes
        let result = rule.apply("\u{201C}hello\u{201D}", Lang::En).unwrap();
        assert_eq!(result, "\"hello\"");
    }

    #[test]
    fn test_symbol_rule() {
        let rule = SymbolRule;

        let result = rule.apply("hello@world", Lang::En).unwrap();
        assert_eq!(result, "hello at world");

        let result = rule.apply("100%", Lang::En).unwrap();
        assert_eq!(result, "100 percent ");
    }
}
