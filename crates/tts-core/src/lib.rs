//! # tts-core
//!
//! Core types, traits, and error definitions for the Qwen3-TTS Rust Engine.
//!
//! This crate provides the foundational abstractions used across all other crates
//! in the workspace, including:
//!
//! - Common data types (`NormText`, `TokenSeq`, `AudioChunk`, etc.)
//! - Trait definitions for pipeline components
//! - Unified error handling via `TtsError`
//! - Configuration structures

pub mod config;
pub mod error;
pub mod traits;
pub mod types;

pub use config::{ModelConfig, RuntimeConfig, ServerConfig};
pub use error::{TtsError, TtsResult};
pub use traits::{AudioCodec, TextNormalizer, TextTokenizer};
pub use types::{AudioChunk, Lang, NormText, Priority, SynthesisRequest, TokenSeq};
