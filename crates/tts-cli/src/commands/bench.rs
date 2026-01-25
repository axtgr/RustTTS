//! Benchmark command implementation.

use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use tracing::info;

/// Run the benchmark command.
pub async fn run(_model_config: &Path, iterations: usize, text: &str) -> Result<()> {
    println!("Running benchmark with {} iterations", iterations);
    println!("Text: \"{text}\"");
    println!();

    // Benchmark text normalization
    let normalizer = text_normalizer::Normalizer::new();
    let norm_times: Vec<f64> = (0..iterations)
        .map(|_| {
            let start = Instant::now();
            let _ =
                tts_core::TextNormalizer::normalize(&normalizer, text, Some(tts_core::Lang::Ru));
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();

    let norm_avg = norm_times.iter().sum::<f64>() / iterations as f64;
    let norm_p95 = percentile(&norm_times, 95);

    println!("Normalization:");
    println!("  Avg: {norm_avg:.3} ms");
    println!("  P95: {norm_p95:.3} ms");
    println!();

    // Benchmark mock tokenization
    let tokenizer = text_tokenizer::MockTokenizer::new(65536);
    let norm_text = tts_core::NormText::new(text, tts_core::Lang::Ru);

    let tok_times: Vec<f64> = (0..iterations)
        .map(|_| {
            let start = Instant::now();
            let _ = tts_core::TextTokenizer::encode(&tokenizer, &norm_text);
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();

    let tok_avg = tok_times.iter().sum::<f64>() / iterations as f64;
    let tok_p95 = percentile(&tok_times, 95);

    println!("Tokenization (mock):");
    println!("  Avg: {tok_avg:.3} ms");
    println!("  P95: {tok_p95:.3} ms");
    println!();

    // Benchmark mock codec decoding
    let codec = audio_codec_12hz::Codec12Hz::new();
    let mock_tokens: Vec<u32> = (0..100).collect();

    let decode_times: Vec<f64> = (0..iterations)
        .map(|_| {
            let start = Instant::now();
            let _ = tts_core::AudioCodec::decode(&codec, &mock_tokens);
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();

    let decode_avg = decode_times.iter().sum::<f64>() / iterations as f64;
    let decode_p95 = percentile(&decode_times, 95);

    println!("Audio decode (mock, 100 tokens):");
    println!("  Avg: {decode_avg:.3} ms");
    println!("  P95: {decode_p95:.3} ms");
    println!();

    info!("Benchmark complete");
    println!("Note: These are mock benchmarks. Real model benchmarks require loaded weights.");

    Ok(())
}

/// Calculate the p-th percentile of a sorted slice.
fn percentile(values: &[f64], p: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = (p as f64 / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
