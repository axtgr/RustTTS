//! # runtime
//!
//! Runtime orchestration for the Qwen3-TTS Rust Engine.
//!
//! This crate provides:
//! - TTS pipeline integration (text → audio)
//! - Streaming synthesis sessions
//! - Request queue management
//! - Batching for efficient GPU utilization
//! - QoS policies (priority, deadlines, cancellation)
//! - Structured logging and metrics
//! - Device selection (CPU, Metal, CUDA)

pub mod allocator;
pub mod device;
pub mod logging;
pub mod metal_cache;
pub mod metrics;
pub mod pipeline;
pub mod profiler;
pub mod queue;
pub mod tracing_setup;
pub mod warm;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use tokio::sync::mpsc;
use tracing::{debug, info, instrument, warn};

use tts_core::{AudioChunk, RuntimeConfig, SynthesisRequest, TtsError, TtsResult};

pub use device::{
    DevicePreference, device_name, is_cuda_available, is_metal_available, select_device,
};
pub use pipeline::{PipelineBackend, PipelineConfig, StreamingSession, TtsPipeline};
pub use queue::{QueuedRequest, RequestQueue};

/// TTS runtime orchestrator.
///
/// Manages the TTS pipeline, request queue, and batch scheduling.
#[derive(Debug)]
pub struct TtsRuntime {
    config: RuntimeConfig,
    pipeline: Arc<TtsPipeline>,
    queue: RequestQueue,
}

impl TtsRuntime {
    /// Create a new TTS runtime with mock backend.
    pub fn new_mock(config: RuntimeConfig) -> TtsResult<Self> {
        info!("Initializing TTS runtime with mock backend");

        let pipeline = Arc::new(TtsPipeline::new_mock()?);
        let queue = RequestQueue::new(config.queue.max_queue_size);

        Ok(Self {
            config,
            pipeline,
            queue,
        })
    }

    /// Create a new TTS runtime from pretrained model directories.
    ///
    /// # Arguments
    /// * `config` - Runtime configuration
    /// * `talker_dir` - Path to Qwen3-TTS talker model directory
    /// * `tokenizer_dir` - Path to tokenizer directory
    /// * `codec_dir` - Path to audio codec directory
    /// * `device` - Device to load models on
    #[instrument(skip_all)]
    pub fn from_pretrained(
        config: RuntimeConfig,
        talker_dir: impl AsRef<Path>,
        tokenizer_dir: impl AsRef<Path>,
        codec_dir: impl AsRef<Path>,
        device: &Device,
    ) -> TtsResult<Self> {
        info!("Initializing TTS runtime from pretrained models");

        let pipeline = Arc::new(TtsPipeline::from_pretrained(
            talker_dir,
            tokenizer_dir,
            codec_dir,
            device,
        )?);
        let queue = RequestQueue::new(config.queue.max_queue_size);

        Ok(Self {
            config,
            pipeline,
            queue,
        })
    }

    /// Create a new TTS runtime with neural backend (legacy API).
    #[instrument(skip(config, _acoustic_path, _tokenizer_path, _codec_path))]
    #[deprecated(since = "0.2.0", note = "Use from_pretrained() instead")]
    #[allow(deprecated)]
    pub fn new_neural(
        config: RuntimeConfig,
        _acoustic_path: impl AsRef<std::path::Path>,
        _tokenizer_path: impl AsRef<std::path::Path>,
        _codec_path: impl AsRef<std::path::Path>,
    ) -> TtsResult<Self> {
        info!("Legacy neural API called, using mock backend");
        Self::new_mock(config)
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get a reference to the pipeline.
    pub fn pipeline(&self) -> &TtsPipeline {
        &self.pipeline
    }

    /// Get an Arc to the pipeline for sharing.
    pub fn pipeline_arc(&self) -> Arc<TtsPipeline> {
        Arc::clone(&self.pipeline)
    }

    /// Get queue statistics.
    pub fn queue_stats(&self) -> QueueStats {
        QueueStats {
            size: self.queue.len(),
            max_size: self.config.queue.max_queue_size,
            is_full: self.queue.is_full(),
        }
    }

    /// Submit a synthesis request for processing.
    ///
    /// Returns a channel receiver for streaming audio chunks.
    #[instrument(skip(self, request), fields(session_id = %request.session_id))]
    pub async fn submit(
        &self,
        request: SynthesisRequest,
    ) -> TtsResult<mpsc::Receiver<TtsResult<AudioChunk>>> {
        let start = Instant::now();

        // Check queue capacity
        if self.queue.is_full() {
            warn!("Request queue is full");
            return Err(TtsError::queue_full());
        }

        // Create response channel
        let (tx, rx) = mpsc::channel(32);

        // Clone pipeline for async processing
        let pipeline = Arc::clone(&self.pipeline);
        let session_id = request.session_id;

        // Spawn synthesis task
        tokio::spawn(async move {
            debug!(session_id = %session_id, "Starting synthesis");

            match pipeline.synthesize(&request.text, Some(request.lang)) {
                Ok(audio) => {
                    let _ = tx.send(Ok(audio)).await;
                    debug!(
                        session_id = %session_id,
                        elapsed_ms = start.elapsed().as_millis(),
                        "Synthesis completed"
                    );
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    warn!(session_id = %session_id, "Synthesis failed");
                }
            }
        });

        Ok(rx)
    }

    /// Synthesize text synchronously (blocking).
    pub fn synthesize_sync(
        &self,
        text: &str,
        lang: Option<tts_core::Lang>,
    ) -> TtsResult<AudioChunk> {
        self.pipeline.synthesize(text, lang)
    }

    /// Create a streaming synthesis session.
    pub fn streaming_session(&self) -> TtsResult<StreamingSession<'_>> {
        self.pipeline.streaming_session()
    }

    /// Cancel a pending request.
    pub fn cancel(&self, session_id: uuid::Uuid) -> bool {
        self.queue.cancel(session_id)
    }

    /// Shutdown the runtime gracefully.
    pub async fn shutdown(&self) -> TtsResult<()> {
        info!("Shutting down TTS runtime");
        self.queue.clear();
        Ok(())
    }
}

impl Default for TtsRuntime {
    fn default() -> Self {
        Self::new_mock(RuntimeConfig::default()).expect("default config should be valid")
    }
}

/// Queue statistics.
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Current queue size.
    pub size: usize,
    /// Maximum queue capacity.
    pub max_size: usize,
    /// Whether the queue is full.
    pub is_full: bool,
}

/// Batch scheduler for efficient processing of multiple requests.
#[derive(Debug)]
pub struct BatchScheduler {
    max_batch_size: usize,
    max_batch_tokens: usize,
    batch_window_ms: u64,
}

impl BatchScheduler {
    /// Create a new batch scheduler.
    pub fn new(max_batch_size: usize, max_batch_tokens: usize, batch_window_ms: u64) -> Self {
        Self {
            max_batch_size,
            max_batch_tokens,
            batch_window_ms,
        }
    }

    /// Create from runtime config.
    pub fn from_config(config: &RuntimeConfig) -> Self {
        Self::new(
            config.batching.max_batch_size,
            config.batching.max_batch_tokens,
            config.batching.batch_window_ms,
        )
    }

    /// Get maximum batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Get maximum tokens per batch.
    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }

    /// Get batch window duration in milliseconds.
    pub fn batch_window_ms(&self) -> u64 {
        self.batch_window_ms
    }

    /// Collect requests into a batch.
    ///
    /// Returns requests that fit within batch constraints.
    pub fn collect_batch(&self, requests: &mut Vec<QueuedRequest>) -> Vec<QueuedRequest> {
        let mut batch = Vec::with_capacity(self.max_batch_size);
        let mut total_tokens = 0;

        while !requests.is_empty() && batch.len() < self.max_batch_size {
            let req = &requests[0];

            // Estimate token count (rough: 1 token per 4 chars)
            let estimated_tokens = req.request.text.len() / 4 + 1;

            if total_tokens + estimated_tokens > self.max_batch_tokens && !batch.is_empty() {
                break;
            }

            batch.push(requests.remove(0));
            total_tokens += estimated_tokens;
        }

        batch
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new(8, 4096, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_core::Lang;

    #[test]
    fn test_runtime_creation_mock() {
        let runtime = TtsRuntime::new_mock(RuntimeConfig::default()).unwrap();
        assert_eq!(runtime.config().batching.max_batch_size, 8);
    }

    #[test]
    fn test_runtime_default() {
        let runtime = TtsRuntime::default();
        assert_eq!(runtime.config().batching.max_batch_size, 8);
    }

    #[test]
    fn test_runtime_queue_stats() {
        let runtime = TtsRuntime::default();
        let stats = runtime.queue_stats();

        assert_eq!(stats.size, 0);
        assert_eq!(stats.max_size, 1000);
        assert!(!stats.is_full);
    }

    #[test]
    fn test_runtime_synthesize_sync() {
        let runtime = TtsRuntime::default();
        let audio = runtime.synthesize_sync("Тест", Some(Lang::Ru)).unwrap();

        assert!(audio.num_samples() > 0);
        assert_eq!(audio.sample_rate, 24000);
    }

    #[tokio::test]
    async fn test_runtime_submit() {
        let runtime = TtsRuntime::default();
        let request = SynthesisRequest::new("Test synthesis").with_lang(Lang::En);

        let mut rx = runtime.submit(request).await.unwrap();

        // Should receive audio chunk
        let result = rx.recv().await;
        assert!(result.is_some());

        let audio = result.unwrap().unwrap();
        assert!(audio.num_samples() > 0);
    }

    #[tokio::test]
    async fn test_runtime_shutdown() {
        let runtime = TtsRuntime::default();
        let result = runtime.shutdown().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_scheduler_creation() {
        let scheduler = BatchScheduler::new(16, 8192, 20);

        assert_eq!(scheduler.max_batch_size(), 16);
        assert_eq!(scheduler.max_batch_tokens(), 8192);
        assert_eq!(scheduler.batch_window_ms(), 20);
    }

    #[test]
    fn test_batch_scheduler_from_config() {
        let config = RuntimeConfig::default();
        let scheduler = BatchScheduler::from_config(&config);

        assert_eq!(scheduler.max_batch_size(), config.batching.max_batch_size);
    }

    #[test]
    fn test_batch_scheduler_collect_empty() {
        let scheduler = BatchScheduler::default();
        let mut requests = Vec::new();

        let batch = scheduler.collect_batch(&mut requests);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_scheduler_collect_batch() {
        let scheduler = BatchScheduler::new(2, 1000, 10);

        let mut requests: Vec<QueuedRequest> = (0..5)
            .map(|i| QueuedRequest::new(SynthesisRequest::new(format!("Request {i}"))))
            .collect();

        let batch = scheduler.collect_batch(&mut requests);

        assert_eq!(batch.len(), 2); // max_batch_size
        assert_eq!(requests.len(), 3); // remaining
    }
}
