//! Normalize command implementation.

use anyhow::{Result, bail};
use text_normalizer::Normalizer;
use tts_core::TextNormalizer;

/// Run the normalize command.
pub fn run(input: &str, lang: &str) -> Result<()> {
    let lang = match lang.to_lowercase().as_str() {
        "ru" => tts_core::Lang::Ru,
        "en" => tts_core::Lang::En,
        "mixed" => tts_core::Lang::Mixed,
        _ => bail!("unknown language: {lang}"),
    };

    let normalizer = Normalizer::new();
    let result = normalizer.normalize(input, Some(lang))?;

    println!("Input:      {input}");
    println!("Normalized: {}", result.text);
    println!("Language:   {}", result.lang);

    if !result.spans.is_empty() {
        println!("Spans:");
        for span in &result.spans {
            println!(
                "  [{}-{}] {:?} ({})",
                span.start, span.end, span.lang, span.kind
            );
        }
    }

    Ok(())
}
