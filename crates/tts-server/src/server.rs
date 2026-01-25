//! TTS Server implementation with gRPC and HTTP endpoints.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use serde::{Deserialize, Serialize};
use tokio::signal;
use tokio::sync::watch;
use tonic::transport::Server as TonicServer;
use tracing::{info, warn};

use runtime::TtsPipeline;
use tts_core::TtsResult;

use crate::proto::tts_service_server::TtsServiceServer;
use crate::service::TtsServiceImpl;

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// gRPC server address.
    pub grpc_addr: SocketAddr,
    /// HTTP server address (for health/metrics).
    pub http_addr: SocketAddr,
    /// Enable Prometheus metrics.
    pub metrics_enabled: bool,
    /// Graceful shutdown timeout in seconds.
    pub shutdown_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            grpc_addr: "0.0.0.0:50051".parse().unwrap(),
            http_addr: "0.0.0.0:8080".parse().unwrap(),
            metrics_enabled: true,
            shutdown_timeout_secs: 30,
        }
    }
}

/// Shared server state.
struct AppState {
    #[allow(dead_code)]
    pipeline: Arc<TtsPipeline>,
    start_time: std::time::Instant,
}

/// Health check response.
#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
}

/// Info response.
#[derive(Serialize)]
struct InfoResponse {
    name: &'static str,
    version: &'static str,
    grpc_addr: String,
    http_addr: String,
}

/// The main TTS server.
pub struct TtsServer {
    config: ServerConfig,
    pipeline: Arc<TtsPipeline>,
}

impl TtsServer {
    /// Create a new TTS server with mock pipeline.
    pub fn new_mock(config: ServerConfig) -> TtsResult<Self> {
        let pipeline = Arc::new(TtsPipeline::new_mock()?);
        Ok(Self { config, pipeline })
    }

    /// Create a new TTS server with custom pipeline.
    pub fn with_pipeline(config: ServerConfig, pipeline: Arc<TtsPipeline>) -> Self {
        Self { config, pipeline }
    }

    /// Run the server.
    pub async fn run(self) -> TtsResult<()> {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // Create gRPC service
        let tts_service = TtsServiceImpl::new(Arc::clone(&self.pipeline));
        let grpc_service = TtsServiceServer::new(tts_service);

        // Create HTTP app state
        let app_state = Arc::new(AppState {
            pipeline: Arc::clone(&self.pipeline),
            start_time: std::time::Instant::now(),
        });

        // Create HTTP router
        let http_app = Router::new()
            .route("/health", get(health_handler))
            .route("/healthz", get(health_handler))
            .route("/ready", get(ready_handler))
            .route("/info", get(info_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(app_state);

        // Spawn HTTP server
        let http_addr = self.config.http_addr;
        let mut http_shutdown_rx = shutdown_rx.clone();
        let http_handle = tokio::spawn(async move {
            info!(addr = %http_addr, "Starting HTTP server");

            let listener = tokio::net::TcpListener::bind(http_addr)
                .await
                .expect("Failed to bind HTTP address");

            axum::serve(listener, http_app)
                .with_graceful_shutdown(async move {
                    http_shutdown_rx.changed().await.ok();
                })
                .await
                .expect("HTTP server failed");
        });

        // Start gRPC server
        let grpc_addr = self.config.grpc_addr;
        let mut grpc_shutdown_rx = shutdown_rx.clone();

        info!(addr = %grpc_addr, "Starting gRPC server");

        let grpc_handle = tokio::spawn(async move {
            TonicServer::builder()
                .add_service(grpc_service)
                .serve_with_shutdown(grpc_addr, async move {
                    grpc_shutdown_rx.changed().await.ok();
                })
                .await
                .expect("gRPC server failed");
        });

        info!(
            grpc = %self.config.grpc_addr,
            http = %self.config.http_addr,
            "TTS server started"
        );

        // Wait for shutdown signal
        shutdown_signal().await;

        info!("Shutdown signal received, stopping servers...");

        // Signal shutdown
        let _ = shutdown_tx.send(true);

        // Wait for servers to stop with timeout
        let timeout = Duration::from_secs(self.config.shutdown_timeout_secs);
        tokio::select! {
            _ = tokio::time::sleep(timeout) => {
                warn!("Shutdown timeout, forcing exit");
            }
            _ = async {
                let _ = http_handle.await;
                let _ = grpc_handle.await;
            } => {
                info!("Servers stopped gracefully");
            }
        }

        Ok(())
    }
}

/// Health check handler.
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();

    Json(HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: uptime,
    })
}

/// Readiness check handler.
async fn ready_handler() -> impl IntoResponse {
    StatusCode::OK
}

/// Info handler.
async fn info_handler() -> impl IntoResponse {
    Json(InfoResponse {
        name: "Qwen3-TTS Rust Server",
        version: env!("CARGO_PKG_VERSION"),
        grpc_addr: "0.0.0.0:50051".to_string(),
        http_addr: "0.0.0.0:8080".to_string(),
    })
}

/// Metrics handler (Prometheus format).
async fn metrics_handler() -> impl IntoResponse {
    // TODO: Integrate with metrics crate
    let metrics = r#"# HELP tts_requests_total Total TTS requests
# TYPE tts_requests_total counter
tts_requests_total 0
# HELP tts_request_duration_ms Request duration in milliseconds
# TYPE tts_request_duration_ms histogram
"#;

    (
        StatusCode::OK,
        [("Content-Type", "text/plain; charset=utf-8")],
        metrics,
    )
}

/// Wait for shutdown signal (SIGINT or SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.grpc_addr.port(), 50051);
        assert_eq!(config.http_addr.port(), 8080);
    }

    #[tokio::test]
    async fn test_server_creation() {
        let config = ServerConfig::default();
        let server = TtsServer::new_mock(config);
        assert!(server.is_ok());
    }
}
