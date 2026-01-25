//! Metrics collection and Prometheus export.

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use tts_core::TtsResult;

/// Metrics recorder for TTS operations.
#[derive(Debug)]
pub struct TtsMetrics;

impl TtsMetrics {
    /// Initialize the metrics system and start Prometheus exporter.
    ///
    /// # Arguments
    /// * `port` - Port for the Prometheus metrics endpoint
    pub fn init(port: u16) -> TtsResult<Self> {
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();

        PrometheusBuilder::new()
            .with_http_listener(addr)
            .install()
            .map_err(|e| tts_core::TtsError::internal(format!("metrics init failed: {e}")))?;

        // Register metric descriptions
        Self::register_metrics();

        Ok(Self)
    }

    /// Initialize metrics without starting an HTTP server (for testing).
    pub fn init_noop() -> Self {
        // Just register descriptions without a recorder
        Self
    }

    fn register_metrics() {
        // Request metrics
        describe_counter!(
            "tts_requests_total",
            "Total number of TTS requests received"
        );
        describe_counter!(
            "tts_requests_completed",
            "Total number of TTS requests completed successfully"
        );
        describe_counter!(
            "tts_requests_failed",
            "Total number of TTS requests that failed"
        );
        describe_counter!(
            "tts_requests_timeout",
            "Total number of TTS requests that timed out"
        );

        // Latency metrics
        describe_histogram!(
            "tts_time_to_first_audio_ms",
            "Time to first audio chunk in milliseconds"
        );
        describe_histogram!(
            "tts_total_latency_ms",
            "Total request latency in milliseconds"
        );
        describe_histogram!(
            "tts_inference_latency_ms",
            "Model inference latency in milliseconds"
        );

        // Throughput metrics
        describe_gauge!("tts_active_requests", "Number of currently active requests");
        describe_gauge!("tts_queue_size", "Current size of the request queue");
        describe_histogram!(
            "tts_rtf",
            "Real-time factor (processing time / audio duration)"
        );

        // Resource metrics
        describe_gauge!("tts_gpu_memory_used_bytes", "GPU memory usage in bytes");
        describe_gauge!("tts_kv_cache_entries", "Number of entries in the KV cache");
    }

    // Request tracking methods

    /// Record a new request received.
    pub fn request_received(&self) {
        counter!("tts_requests_total").increment(1);
    }

    /// Record a request completed successfully.
    pub fn request_completed(&self) {
        counter!("tts_requests_completed").increment(1);
    }

    /// Record a request failed.
    pub fn request_failed(&self) {
        counter!("tts_requests_failed").increment(1);
    }

    /// Record a request timeout.
    pub fn request_timeout(&self) {
        counter!("tts_requests_timeout").increment(1);
    }

    // Latency tracking methods

    /// Record time to first audio.
    pub fn record_time_to_first_audio(&self, ms: f64) {
        histogram!("tts_time_to_first_audio_ms").record(ms);
    }

    /// Record total latency.
    pub fn record_total_latency(&self, ms: f64) {
        histogram!("tts_total_latency_ms").record(ms);
    }

    /// Record inference latency.
    pub fn record_inference_latency(&self, ms: f64) {
        histogram!("tts_inference_latency_ms").record(ms);
    }

    /// Record real-time factor.
    pub fn record_rtf(&self, rtf: f64) {
        histogram!("tts_rtf").record(rtf);
    }

    // Resource tracking methods

    /// Set the number of active requests.
    pub fn set_active_requests(&self, count: f64) {
        gauge!("tts_active_requests").set(count);
    }

    /// Set the queue size.
    pub fn set_queue_size(&self, size: f64) {
        gauge!("tts_queue_size").set(size);
    }

    /// Set GPU memory usage.
    pub fn set_gpu_memory(&self, bytes: f64) {
        gauge!("tts_gpu_memory_used_bytes").set(bytes);
    }

    /// Set KV cache entry count.
    pub fn set_kv_cache_entries(&self, count: f64) {
        gauge!("tts_kv_cache_entries").set(count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_noop() {
        let metrics = TtsMetrics::init_noop();

        // These should not panic even without a recorder
        metrics.request_received();
        metrics.request_completed();
        metrics.record_time_to_first_audio(100.0);
        metrics.set_active_requests(5.0);
    }
}
