//! Synthesis command implementation.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use tracing::{debug, info};

use audio_codec_12hz::wav::{DEFAULT_FADE_IN_MS, apply_fade_in, write_wav};
use runtime::{
    TtsPipeline,
    device::{DevicePreference, select_device},
};
use tts_core::Lang;

/// Options for synthesis command.
pub struct SynthOptions {
    pub input: String,
    pub output: PathBuf,
    pub lang: String,
    /// Speaker name for CustomVoice models (e.g., "vivian", "ryan").
    pub speaker: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub codec_dir: Option<PathBuf>,
    /// Legacy model config path (reserved for future use).
    #[allow(dead_code)]
    pub model_config: Option<PathBuf>,
    /// Random seed for reproducible generation (reserved for future use).
    #[allow(dead_code)]
    pub seed: Option<u64>,
    /// Use multi-codebook decoding (all 16 codebooks via CodePredictor).
    pub multi_codebook: bool,
    /// Device preference (Auto, Cpu, Metal, Cuda).
    pub device_preference: DevicePreference,
}

/// Create pipeline based on options.
fn create_pipeline(options: &SynthOptions) -> Result<TtsPipeline> {
    if let Some(ref model_dir) = options.model_dir {
        info!(model_dir = %model_dir.display(), "Loading pretrained model");

        // Use model_dir for both talker and tokenizer
        let tokenizer_dir = model_dir;

        // Codec dir - use provided or default
        let codec_dir = options.codec_dir.clone().unwrap_or_else(|| {
            // 1. Try finding speech_tokenizer inside model_dir (bundled codec)
            let bundled = model_dir.join("speech_tokenizer");
            if bundled.exists() {
                debug!(path = %bundled.display(), "Found bundled speech_tokenizer");
                return bundled;
            }

            // 2. Try to find codec in parent directory
            model_dir
                .parent()
                .map(|p| p.join("qwen3-tts-tokenizer"))
                .unwrap_or_else(|| PathBuf::from("models/qwen3-tts-tokenizer"))
        });

        // Ensure model.safetensors exists to avoid silent fallback to Mock
        if !model_dir.join("model.safetensors").exists() {
            bail!(
                "Acoustic model weights (model.safetensors) not found in directory: {}",
                model_dir.display()
            );
        }

        let device = select_device(options.device_preference)?;

        let pipeline = TtsPipeline::from_pretrained(model_dir, tokenizer_dir, &codec_dir, &device)?;

        Ok(pipeline)
    } else {
        info!("Using mock pipeline (no --model-dir specified)");
        Ok(TtsPipeline::new_mock()?)
    }
}

/// Run the synthesis command.
pub async fn run(options: SynthOptions) -> Result<()> {
    let start = Instant::now();

    // Parse language
    let lang = match options.lang.to_lowercase().as_str() {
        "ru" => Lang::Ru,
        "en" => Lang::En,
        "mixed" => Lang::Mixed,
        _ => bail!(
            "unknown language: {}, expected: ru, en, or mixed",
            options.lang
        ),
    };

    // Get input text
    let text = if let Some(path) = options.input.strip_prefix('@') {
        info!(path = path, "Reading text from file");
        std::fs::read_to_string(path)?
    } else {
        options.input.clone()
    };

    if text.trim().is_empty() {
        bail!("input text is empty");
    }

    info!(
        text_len = text.len(),
        lang = %lang,
        output = %options.output.display(),
        speaker = ?options.speaker,
        multi_codebook = options.multi_codebook,
        "Starting synthesis"
    );

    // Create pipeline
    let pipeline = create_pipeline(&options)?;

    // Synthesize (with optional speaker for CustomVoice models)
    let synth_start = Instant::now();
    let audio = if options.multi_codebook {
        info!("Using multi-codebook decoding (all 16 codebooks)");
        pipeline.synthesize_with_multi_codebook(&text, Some(lang), options.speaker.as_deref())?
    } else {
        pipeline.synthesize_with_speaker(&text, Some(lang), options.speaker.as_deref())?
    };
    let synth_duration = synth_start.elapsed();

    debug!(
        samples = audio.num_samples(),
        sample_rate = audio.sample_rate,
        duration_ms = audio.duration_ms(),
        synth_ms = synth_duration.as_millis(),
        "Synthesis completed"
    );

    // Apply fade-in to remove initial artifacts
    let mut samples: Vec<f32> = audio.pcm.to_vec();
    apply_fade_in(&mut samples, DEFAULT_FADE_IN_MS, audio.sample_rate);
    let audio = tts_core::AudioChunk::new(samples, audio.sample_rate, audio.start_ms, audio.end_ms);

    // Calculate real-time factor
    let audio_duration_sec = audio.duration_ms() / 1000.0;
    let process_sec = synth_duration.as_secs_f32();
    let rtf = if audio_duration_sec > 0.0 {
        process_sec / audio_duration_sec
    } else {
        0.0
    };

    // Write to WAV file
    write_wav(&options.output, &audio)?;

    let total_duration = start.elapsed();

    // Print summary
    println!("Synthesis complete!");
    println!();
    println!("Input:     {} chars", text.len());
    println!("Language:  {}", lang);
    println!("Output:    {}", options.output.display());
    println!();
    println!("Audio:");
    println!("  Duration:    {:.2} sec", audio_duration_sec);
    println!("  Samples:     {}", audio.num_samples());
    println!("  Sample rate: {} Hz", audio.sample_rate);
    println!();
    println!("Performance:");
    println!("  Synthesis:   {:.1} ms", synth_duration.as_millis());
    println!("  Total:       {:.1} ms", total_duration.as_millis());
    println!("  RTF:         {:.3}x", rtf);

    if rtf < 1.0 {
        println!("  Status:      Faster than real-time!");
    } else {
        println!("  Status:      Slower than real-time");
    }

    info!(
        output = %options.output.display(),
        duration_ms = audio.duration_ms(),
        rtf = rtf,
        "Synthesis saved to file"
    );

    Ok(())
}

/// Run streaming synthesis command.
pub async fn run_streaming(options: SynthOptions) -> Result<()> {
    let start = Instant::now();

    // Parse language
    let lang = match options.lang.to_lowercase().as_str() {
        "ru" => Lang::Ru,
        "en" => Lang::En,
        "mixed" => Lang::Mixed,
        _ => bail!("unknown language: {}", options.lang),
    };

    // Get input text
    let text = if let Some(path) = options.input.strip_prefix('@') {
        std::fs::read_to_string(path)?
    } else {
        options.input.clone()
    };

    if text.trim().is_empty() {
        bail!("input text is empty");
    }

    info!(
        text_len = text.len(),
        output = %options.output.display(),
        "Starting streaming synthesis"
    );

    // Create pipeline and streaming session
    let pipeline = create_pipeline(&options)?;
    let mut session = pipeline.streaming_session()?;

    session.set_text(&text, Some(lang))?;

    // Collect all audio chunks
    let mut all_samples: Vec<f32> = Vec::new();
    let mut chunk_count = 0;

    while let Some(chunk) = session.next_chunk()? {
        chunk_count += 1;
        all_samples.extend_from_slice(&chunk.pcm);

        debug!(
            chunk = chunk_count,
            samples = chunk.num_samples(),
            total_samples = all_samples.len(),
            "Received chunk"
        );

        if session.is_finished() {
            break;
        }
    }

    // Apply fade-in to remove initial artifacts
    apply_fade_in(&mut all_samples, DEFAULT_FADE_IN_MS, 24000);

    // Write combined audio to WAV
    let audio = tts_core::AudioChunk::new(all_samples, 24000, 0.0, 0.0);
    write_wav(&options.output, &audio)?;

    let total_duration = start.elapsed();

    println!("Streaming synthesis complete!");
    println!("  Chunks:      {}", chunk_count);
    println!("  Samples:     {}", audio.num_samples());
    println!("  Total time:  {:.1} ms", total_duration.as_millis());
    println!("  Output:      {}", options.output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_options(input: &str, output: PathBuf, lang: &str) -> SynthOptions {
        SynthOptions {
            input: input.to_string(),
            output,
            lang: lang.to_string(),
            speaker: None,
            model_dir: None,
            codec_dir: None,
            model_config: None,
            seed: None,
            multi_codebook: false,
            device_preference: DevicePreference::Auto,
        }
    }

    #[tokio::test]
    async fn test_synth_basic() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test.wav");

        let options = make_options("Hello world", output.clone(), "en");
        let result = run(options).await;

        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[tokio::test]
    async fn test_synth_russian() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test_ru.wav");

        let options = make_options("Привет мир", output.clone(), "ru");
        let result = run(options).await;

        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[tokio::test]
    async fn test_synth_empty_error() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test.wav");

        let options = make_options("", output, "en");
        let result = run(options).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_synth_invalid_lang() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test.wav");

        let options = make_options("Test", output, "invalid");
        let result = run(options).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_streaming_synth() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("stream.wav");

        let options = make_options("Test streaming", output.clone(), "en");
        let result = run_streaming(options).await;

        assert!(result.is_ok());
        assert!(output.exists());
    }
}
