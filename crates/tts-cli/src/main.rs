//! Qwen3-TTS command-line interface.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::info;

mod commands;

/// Qwen3-TTS Rust Engine CLI
#[derive(Debug, Parser)]
#[command(name = "tts")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Log level
    #[arg(short, long, default_value = "info", global = true)]
    log_level: String,

    /// Log format (json or text)
    #[arg(long, default_value = "text", global = true)]
    log_format: LogFormatArg,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LogFormatArg {
    Json,
    Text,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Synthesize text to audio
    Synth {
        /// Input text or file path (use @file.txt for file input)
        input: String,

        /// Output file path (WAV format)
        #[arg(short, long)]
        output: PathBuf,

        /// Language hint (ru, en, or mixed)
        #[arg(long, default_value = "ru")]
        lang: String,

        /// Speaker ID (if model supports multiple speakers)
        #[arg(short, long)]
        speaker: Option<u32>,

        /// Model configuration file
        #[arg(short, long)]
        model_config: Option<PathBuf>,

        /// Random seed for deterministic generation
        #[arg(long)]
        seed: Option<u64>,

        /// Use streaming mode
        #[arg(long)]
        streaming: bool,
    },

    /// Normalize text without synthesis (dry run)
    Normalize {
        /// Input text
        input: String,

        /// Language hint (ru, en, or mixed)
        #[arg(long, default_value = "ru")]
        lang: String,
    },

    /// Tokenize text (dry run)
    Tokenize {
        /// Input text
        input: String,

        /// Tokenizer model path
        #[arg(short, long)]
        tokenizer: PathBuf,
    },

    /// Run local benchmark
    Bench {
        /// Model configuration file
        #[arg(short, long)]
        model_config: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,

        /// Test text
        #[arg(short, long, default_value = "Привет, мир!")]
        text: String,
    },

    /// Show version and configuration info
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let format = match cli.log_format {
        LogFormatArg::Json => runtime::logging::LogFormat::Json,
        LogFormatArg::Text => runtime::logging::LogFormat::Text,
    };
    runtime::logging::init_logging(&cli.log_level, format);

    info!(version = env!("CARGO_PKG_VERSION"), "Starting TTS CLI");

    match cli.command {
        Commands::Synth {
            input,
            output,
            lang,
            speaker,
            model_config,
            seed,
            streaming,
        } => {
            if streaming {
                commands::synth::run_streaming(input, output, lang, speaker, model_config, seed)
                    .await
                    .context("streaming synthesis failed")?;
            } else {
                commands::synth::run(input, output, lang, speaker, model_config, seed)
                    .await
                    .context("synthesis failed")?;
            }
        }
        Commands::Normalize { input, lang } => {
            commands::normalize::run(&input, &lang).context("normalization failed")?;
        }
        Commands::Tokenize { input, tokenizer } => {
            commands::tokenize::run(&input, &tokenizer).context("tokenization failed")?;
        }
        Commands::Bench {
            model_config,
            iterations,
            text,
        } => {
            commands::bench::run(&model_config, iterations, &text)
                .await
                .context("benchmark failed")?;
        }
        Commands::Info => {
            commands::info::run();
        }
    }

    Ok(())
}
