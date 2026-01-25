//! Synthesis command implementation.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use tracing::{debug, info};

use audio_codec_12hz::wav::write_wav;
use runtime::TtsPipeline;
use tts_core::Lang;

/// Run the synthesis command.
pub async fn run(
    input: String,
    output: PathBuf,
    lang: String,
    speaker: Option<u32>,
    _model_config: Option<PathBuf>,
    _seed: Option<u64>,
) -> Result<()> {
    let start = Instant::now();

    // Parse language
    let lang = match lang.to_lowercase().as_str() {
        "ru" => Lang::Ru,
        "en" => Lang::En,
        "mixed" => Lang::Mixed,
        _ => bail!("unknown language: {lang}, expected: ru, en, or mixed"),
    };

    // Get input text
    let text = if let Some(path) = input.strip_prefix('@') {
        info!(path = path, "Reading text from file");
        std::fs::read_to_string(path)?
    } else {
        input
    };

    if text.trim().is_empty() {
        bail!("input text is empty");
    }

    info!(
        text_len = text.len(),
        lang = %lang,
        output = %output.display(),
        speaker = ?speaker,
        "Starting synthesis"
    );

    // Create pipeline
    let pipeline = TtsPipeline::new_mock()?;

    // Synthesize
    let synth_start = Instant::now();
    let audio = pipeline.synthesize(&text, Some(lang))?;
    let synth_duration = synth_start.elapsed();

    debug!(
        samples = audio.num_samples(),
        sample_rate = audio.sample_rate,
        duration_ms = audio.duration_ms(),
        synth_ms = synth_duration.as_millis(),
        "Synthesis completed"
    );

    // Calculate real-time factor
    let audio_duration_sec = audio.duration_ms() / 1000.0;
    let process_sec = synth_duration.as_secs_f32();
    let rtf = if audio_duration_sec > 0.0 {
        process_sec / audio_duration_sec
    } else {
        0.0
    };

    // Write to WAV file
    write_wav(&output, &audio)?;

    let total_duration = start.elapsed();

    // Print summary
    println!("Synthesis complete!");
    println!();
    println!("Input:     {} chars", text.len());
    println!("Language:  {}", lang);
    println!("Output:    {}", output.display());
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
        output = %output.display(),
        duration_ms = audio.duration_ms(),
        rtf = rtf,
        "Synthesis saved to file"
    );

    Ok(())
}

/// Run streaming synthesis command.
pub async fn run_streaming(
    input: String,
    output: PathBuf,
    lang: String,
    _speaker: Option<u32>,
    _model_config: Option<PathBuf>,
    _seed: Option<u64>,
) -> Result<()> {
    let start = Instant::now();

    // Parse language
    let lang = match lang.to_lowercase().as_str() {
        "ru" => Lang::Ru,
        "en" => Lang::En,
        "mixed" => Lang::Mixed,
        _ => bail!("unknown language: {lang}"),
    };

    // Get input text
    let text = if let Some(path) = input.strip_prefix('@') {
        std::fs::read_to_string(path)?
    } else {
        input
    };

    if text.trim().is_empty() {
        bail!("input text is empty");
    }

    info!(
        text_len = text.len(),
        output = %output.display(),
        "Starting streaming synthesis"
    );

    // Create pipeline and streaming session
    let pipeline = TtsPipeline::new_mock()?;
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

    // Write combined audio to WAV
    let audio = tts_core::AudioChunk::new(all_samples, 24000, 0.0, 0.0);
    write_wav(&output, &audio)?;

    let total_duration = start.elapsed();

    println!("Streaming synthesis complete!");
    println!("  Chunks:      {}", chunk_count);
    println!("  Samples:     {}", audio.num_samples());
    println!("  Total time:  {:.1} ms", total_duration.as_millis());
    println!("  Output:      {}", output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_synth_basic() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test.wav");

        let result = run(
            "Hello world".to_string(),
            output.clone(),
            "en".to_string(),
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[tokio::test]
    async fn test_synth_russian() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test_ru.wav");

        let result = run(
            "Привет мир".to_string(),
            output.clone(),
            "ru".to_string(),
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[tokio::test]
    async fn test_synth_empty_error() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test.wav");

        let result = run("".to_string(), output, "en".to_string(), None, None, None).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_synth_invalid_lang() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("test.wav");

        let result = run(
            "Test".to_string(),
            output,
            "invalid".to_string(),
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_streaming_synth() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("stream.wav");

        let result = run_streaming(
            "Test streaming".to_string(),
            output.clone(),
            "en".to_string(),
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_ok());
        assert!(output.exists());
    }
}
