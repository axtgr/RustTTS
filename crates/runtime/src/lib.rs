//! # runtime
//!
//! Runtime orchestration for the Qwen3-TTS Rust Engine.
//!
//! This crate provides:
//! - Request queue management
//! - Batching for efficient GPU utilization
//! - QoS policies (priority, deadlines, cancellation)
//! - Structured logging and metrics
//! - Device management (CPU/GPU selection, warmup)

pub mod logging;
pub mod metrics;
pub mod queue;

use tracing::info;
use tts_core::{RuntimeConfig, SynthesisRequest, TtsResult};

/// TTS runtime orchestrator.
#[derive(Debug)]
pub struct TtsRuntime {
    config: RuntimeConfig,
}

impl TtsRuntime {
    /// Create a new TTS runtime with the given configuration.
    pub fn new(config: RuntimeConfig) -> TtsResult<Self> {
        info!("Initializing TTS runtime");
        Ok(Self { config })
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Submit a synthesis request.
    ///
    /// Note: Placeholder for Phase 4 implementation.
    pub async fn submit(&self, _request: SynthesisRequest) -> TtsResult<()> {
        // TODO: Implement in Phase 4
        Ok(())
    }

    /// Shutdown the runtime gracefully.
    pub async fn shutdown(&self) -> TtsResult<()> {
        info!("Shutting down TTS runtime");
        Ok(())
    }
}

impl Default for TtsRuntime {
    fn default() -> Self {
        Self::new(RuntimeConfig::default()).expect("default config should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = TtsRuntime::default();
        assert_eq!(runtime.config().batching.max_batch_size, 8);
    }

    #[tokio::test]
    async fn test_runtime_shutdown() {
        let runtime = TtsRuntime::default();
        let result = runtime.shutdown().await;
        assert!(result.is_ok());
    }
}
