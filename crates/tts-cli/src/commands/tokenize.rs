//! Tokenize command implementation.

use anyhow::Result;
use std::path::Path;
use text_tokenizer::Tokenizer;
use tts_core::{Lang, NormText, TextTokenizer};

/// Run the tokenize command.
pub fn run(input: &str, tokenizer_path: &Path) -> Result<()> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    let norm_text = NormText::new(input, Lang::Mixed);
    let tokens = tokenizer.encode(&norm_text)?;

    println!("Input: {input}");
    println!("Tokens: {} total", tokens.len());
    println!("IDs: {:?}", tokens.ids);

    // Show token details
    println!("\nToken details:");
    for (i, (&id, &(start, end))) in tokens.ids.iter().zip(tokens.offsets.iter()).enumerate() {
        let substr = if end <= input.len() {
            &input[start..end]
        } else {
            "<special>"
        };
        println!("  {i}: id={id}, offset=[{start}:{end}], text=\"{substr}\"");
    }

    // Show special tokens
    println!("\nSpecial tokens:");
    println!("  BOS: {:?}", tokenizer.bos_token_id());
    println!("  EOS: {:?}", tokenizer.eos_token_id());
    println!("  PAD: {:?}", tokenizer.pad_token_id());
    println!("  Vocab size: {}", tokenizer.vocab_size());

    Ok(())
}
