//! Structured logging setup with tracing.

use tracing_subscriber::{
    EnvFilter,
    fmt::{self, format::FmtSpan},
    prelude::*,
};

/// Logging format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// Human-readable text format.
    Text,
    /// JSON format for structured logging.
    #[default]
    Json,
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" | "pretty" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            _ => Err(format!("unknown log format: {s}")),
        }
    }
}

/// Initialize the logging subsystem.
///
/// # Arguments
/// * `level` - Log level filter (e.g., "info", "debug", "trace")
/// * `format` - Output format (text or JSON)
///
/// # Example
/// ```ignore
/// use runtime::logging::{init_logging, LogFormat};
/// init_logging("info", LogFormat::Json);
/// ```
pub fn init_logging(level: &str, format: LogFormat) {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    match format {
        LogFormat::Text => {
            let subscriber = tracing_subscriber::registry().with(env_filter).with(
                fmt::layer()
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_span_events(FmtSpan::CLOSE),
            );

            let _ = tracing::subscriber::set_global_default(subscriber);
        }
        LogFormat::Json => {
            let subscriber = tracing_subscriber::registry().with(env_filter).with(
                fmt::layer()
                    .json()
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_span_events(FmtSpan::CLOSE),
            );

            let _ = tracing::subscriber::set_global_default(subscriber);
        }
    }
}

/// Initialize logging from environment variables.
///
/// Uses:
/// - `RUST_LOG` for log level (default: "info")
/// - `LOG_FORMAT` for format (default: "json")
pub fn init_logging_from_env() {
    let level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let format: LogFormat = std::env::var("LOG_FORMAT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();

    init_logging(&level, format);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_format_parsing() {
        assert_eq!("json".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert_eq!("text".parse::<LogFormat>().unwrap(), LogFormat::Text);
        assert_eq!("pretty".parse::<LogFormat>().unwrap(), LogFormat::Text);
        assert!("invalid".parse::<LogFormat>().is_err());
    }

    #[test]
    fn test_log_format_default() {
        assert_eq!(LogFormat::default(), LogFormat::Json);
    }
}
