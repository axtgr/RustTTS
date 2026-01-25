# AGENTS.md — Guidelines for AI Coding Agents

This document provides instructions for AI agents working on the Qwen3-TTS Rust Engine codebase.

## Project Overview

Pure Rust TTS engine compatible with Qwen3-TTS. **No Python/Torch dependencies allowed.**
Key crates: `tts-core`, `text-normalizer`, `text-tokenizer`, `acoustic-model`, `audio-codec-12hz`, `runtime`, `tts-cli`.

## Build Commands

```bash
# Full workspace build
cargo build --workspace --release

# Build specific crate
cargo build -p tts-core
cargo build -p acoustic-model --features cuda

# Check without building (faster)
cargo check --workspace
```

## Test Commands

```bash
# Run all tests
cargo test --workspace

# Run single test by name
cargo test test_normalize_numbers
cargo test -p text-normalizer test_normalize_numbers

# Run tests in specific crate
cargo test -p tts-core
cargo test -p text-tokenizer

# Run tests with output
cargo test -- --nocapture

# Run golden tests (require model weights)
cargo test --features golden_tests

# Run ignored/expensive tests
cargo test -- --ignored
```

## Linting & Formatting

```bash
# Format code (MUST pass before commit)
cargo fmt --all

# Check formatting without changes
cargo fmt --all -- --check

# Clippy lints (MUST pass with no warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Clippy with all features
cargo clippy --workspace --all-features -- -D warnings
```

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench latency
cargo bench -p acoustic-model
```

## Code Style Guidelines

### Language
- All code comments in English
- All documentation (rustdoc) in English
- Variable/function names in English
- Log messages in English

### Imports Order
```rust
// 1. Standard library
use std::collections::HashMap;
use std::sync::Arc;

// 2. External crates
use anyhow::{Context, Result};
use tracing::{info, instrument};

// 3. Workspace crates
use tts_core::{AudioChunk, TtsError};

// 4. Local modules
use crate::normalizer::Rule;
use super::utils;
```

### Naming Conventions
- Types/Traits: `PascalCase` — `AudioChunk`, `TextNormalizer`
- Functions/methods: `snake_case` — `decode_chunk`, `generate_stream`
- Constants: `SCREAMING_SNAKE_CASE` — `MAX_BATCH_SIZE`, `DEFAULT_CHUNK_MS`
- Modules: `snake_case` — `text_normalizer`, `kv_cache`
- Feature flags: `snake_case` — `cuda`, `golden_tests`, `server`

### Type Annotations
- Prefer explicit types for public APIs
- Use `impl Trait` for return types when appropriate
- Avoid `Box<dyn Trait>` unless necessary for object safety

### Error Handling
- Use `thiserror` for library error types
- Use `anyhow` only in binaries (`tts-cli`, `tts-server`)
- Always provide context with `.context()` or custom error variants
- Never use `.unwrap()` in library code (use `.expect()` with message if truly unreachable)

```rust
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("tokenization failed: {0}")]
    Tokenization(String),
    #[error("model load failed: {path}")]
    ModelLoad { path: PathBuf, source: std::io::Error },
    #[error("decode timeout after {ms}ms")]
    Timeout { ms: u64 },
}
```

### Async & Streaming
- Use `tokio` for async runtime
- Use `futures::Stream` for streaming APIs
- Prefer `async fn` over manual `Future` impl
- Use `#[instrument]` from `tracing` for async functions

### Performance Considerations
- Minimize allocations in hot paths (use `&str`, `&[T]`, `Arc<[T]>`)
- Use `#[inline]` for small, frequently called functions
- Prefer `Vec::with_capacity()` when size is known
- Use `rayon` for CPU parallelism where beneficial

### Documentation
- All public items MUST have rustdoc comments
- Include examples for complex APIs
- Document panics, errors, and safety requirements

### Testing
- Unit tests in same file (`#[cfg(test)]` module)
- Integration tests in `tests/` directory
- Golden tests gated by `#[cfg(feature = "golden_tests")]`
- Use `proptest` for property-based testing where applicable

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_numbers() {
        let norm = Normalizer::new();
        assert_eq!(norm.normalize("100"), "сто");
    }

    #[test]
    #[ignore] // expensive, run with --ignored
    fn test_long_stream_stability() {
        // 10+ minute stream test
    }
}
```

### Feature Flags
```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda"]
server = ["tonic", "prost"]
golden_tests = []
bench = ["criterion"]
```

### Logging
- Use `tracing` crate (not `log`)
- Include structured fields for correlation
- Use appropriate levels: `error` > `warn` > `info` > `debug` > `trace`

```rust
use tracing::{info, debug, instrument};

#[instrument(skip(self), fields(session_id = %req.session_id))]
pub async fn synthesize(&self, req: SynthesisRequest) -> Result<()> {
    info!(text_len = req.text.len(), "starting synthesis");
    debug!(chunk_idx = idx, latency_ms = lat, "chunk emitted");
}
```

## Architecture Reminders
- Tensor backend: prefer `candle`, abstract via `TensorBackend` trait
- KV cache: ring buffer, LRU eviction, device-resident
- Streaming: 20-100ms chunks, Hann window overlap-add (5-10ms)
- Configs: `model.toml`, `runtime.toml`, `server.toml`
