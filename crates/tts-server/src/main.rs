//! Qwen3-TTS gRPC Server.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;

use tts_server::{ServerConfig, TtsServer};

/// Qwen3-TTS gRPC Server
#[derive(Debug, Parser)]
#[command(name = "tts-server")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// gRPC server address
    #[arg(long, default_value = "0.0.0.0:50051")]
    grpc_addr: SocketAddr,

    /// HTTP server address (health/metrics)
    #[arg(long, default_value = "0.0.0.0:8080")]
    http_addr: SocketAddr,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable JSON logging
    #[arg(long)]
    json_logs: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let format = if args.json_logs {
        runtime::logging::LogFormat::Json
    } else {
        runtime::logging::LogFormat::Text
    };
    runtime::logging::init_logging(&args.log_level, format);

    info!(
        version = env!("CARGO_PKG_VERSION"),
        grpc_addr = %args.grpc_addr,
        http_addr = %args.http_addr,
        "Starting TTS server"
    );

    // Create server config
    let config = ServerConfig {
        grpc_addr: args.grpc_addr,
        http_addr: args.http_addr,
        metrics_enabled: true,
        shutdown_timeout_secs: 30,
    };

    // Create and run server
    let server = TtsServer::new_mock(config).context("Failed to create server")?;

    server.run().await.context("Server failed")?;

    info!("Server shutdown complete");
    Ok(())
}
