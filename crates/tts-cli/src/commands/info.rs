//! Info command implementation.

/// Run the info command.
pub fn run() {
    println!("Qwen3-TTS Rust Engine");
    println!("=====================");
    println!();
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Build info:");
    println!("  Rust version: {}", env!("CARGO_PKG_RUST_VERSION"));

    #[cfg(feature = "cuda")]
    println!("  CUDA: enabled");
    #[cfg(not(feature = "cuda"))]
    println!("  CUDA: disabled");

    println!();
    println!("Crates:");
    println!("  tts-core: Core types and traits");
    println!("  text-normalizer: Text normalization (RU/EN)");
    println!("  text-tokenizer: BPE/Unigram tokenization");
    println!("  acoustic-model: Transformer acoustic model");
    println!("  audio-codec-12hz: Audio codec decoder");
    println!("  runtime: Orchestration and batching");
    println!("  tts-cli: This CLI tool");
    println!();
    println!("For more information, see: https://github.com/example/qwen3-tts-rust");
}
