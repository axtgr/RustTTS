//! # tts-server
//!
//! gRPC server for Qwen3-TTS.
//!
//! Provides:
//! - Unary synthesis endpoint
//! - Streaming synthesis endpoint
//! - Health check endpoint
//! - Prometheus metrics endpoint

pub mod proto;
pub mod server;
pub mod service;

pub use server::{ServerConfig, TtsServer};
pub use service::TtsServiceImpl;
