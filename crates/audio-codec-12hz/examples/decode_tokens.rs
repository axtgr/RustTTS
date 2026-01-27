//! Decode tokens from JSON file for comparison with Python.
//!
//! Usage:
//!   cargo run -p audio-codec-12hz --example decode_tokens -- \
//!     --tokens /tmp/tokens_for_rust.json \
//!     --codec models/qwen3-tts-tokenizer \
//!     --output /tmp/rust_decoded_tokens.wav

use std::fs;
use std::path::PathBuf;

use audio_codec_12hz::{Codec12Hz, wav};
use candle_core::Device;
use tts_core::{AudioChunk, TtsResult};

fn main() -> TtsResult<()> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();

    let mut tokens_path = PathBuf::from("/tmp/tokens_for_rust.json");
    let mut codec_path = PathBuf::from("models/qwen3-tts-tokenizer");
    let mut output_path = PathBuf::from("/tmp/rust_decoded_tokens.wav");
    let mut no_smooth = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" => {
                tokens_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--codec" => {
                codec_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" | "-o" => {
                output_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--no-smooth" => {
                no_smooth = true;
                i += 1;
            }
            _ => i += 1,
        }
    }

    println!("Loading tokens from: {}", tokens_path.display());
    println!("Loading codec from: {}", codec_path.display());
    println!("Output: {}", output_path.display());
    println!("Smoothing: {}", !no_smooth);

    // Load tokens from JSON
    // Format: [[token0_cb0, token0_cb1, ..., token0_cb15], [token1_cb0, ...], ...]
    // i.e., [num_frames][num_codebooks]
    let json_str = fs::read_to_string(&tokens_path)
        .map_err(|e| tts_core::TtsError::internal(format!("Failed to read tokens: {e}")))?;
    let frames: Vec<Vec<i64>> = serde_json::from_str(&json_str)
        .map_err(|e| tts_core::TtsError::internal(format!("Failed to parse JSON: {e}")))?;

    let num_frames = frames.len();
    let num_codebooks = frames.get(0).map(|f| f.len()).unwrap_or(0);
    println!("Loaded {} frames x {} codebooks", num_frames, num_codebooks);

    if num_frames == 0 || num_codebooks == 0 {
        return Err(tts_core::TtsError::invalid_input("Empty token data"));
    }

    // Transpose to [num_codebooks][num_frames] for decoder
    let mut tokens: Vec<Vec<u32>> = vec![vec![0u32; num_frames]; num_codebooks];
    for (frame_idx, frame) in frames.iter().enumerate() {
        for (cb_idx, &token) in frame.iter().enumerate() {
            tokens[cb_idx][frame_idx] = token as u32;
        }
    }

    println!(
        "First codebook (first 5): {:?}",
        &tokens[0][..5.min(num_frames)]
    );
    println!(
        "Last codebook (first 5): {:?}",
        &tokens[num_codebooks - 1][..5.min(num_frames)]
    );

    // Load codec
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).expect("Failed to create Metal device");
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    println!("Device: {:?}", device);

    let codec = Codec12Hz::from_pretrained(&codec_path, &device)?;
    println!("Codec loaded");

    // Decode
    println!("Decoding...");
    let audio = codec.decode_multi(&tokens)?;

    let mut pcm = audio.pcm.to_vec();
    println!(
        "Decoded {} samples ({:.3}s)",
        pcm.len(),
        pcm.len() as f32 / 24000.0
    );

    // Analyze before smoothing
    analyze_audio("Before smoothing", &pcm);

    // Apply smoothing if enabled
    if !no_smooth {
        wav::smooth_frame_boundaries_default(&mut pcm);
        wav::smooth_silence_transitions_default(&mut pcm, 24000);
        println!("\nAfter smoothing:");
        analyze_audio("After smoothing", &pcm);
    }

    // Save
    wav::write_wav_samples(&output_path, &pcm, 24000)?;
    println!("\nSaved to {}", output_path.display());

    // Also save without smoothing for comparison
    if !no_smooth {
        let raw_output = output_path.with_file_name(format!(
            "{}_raw.wav",
            output_path.file_stem().unwrap().to_str().unwrap()
        ));
        let raw_audio = codec.decode_multi(&tokens)?;
        wav::write_wav_samples(&raw_output, &raw_audio.pcm, 24000)?;
        println!("Saved raw to {}", raw_output.display());
    }

    Ok(())
}

fn analyze_audio(label: &str, pcm: &[f32]) {
    let rms: f32 = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
    let max_abs: f32 = pcm.iter().map(|x| x.abs()).fold(0.0, f32::max);

    println!("  {}: RMS={:.4}, Max={:.4}", label, rms, max_abs);
    println!("  First 5: {:?}", &pcm[..5.min(pcm.len())]);
    println!("  Last 5: {:?}", &pcm[pcm.len().saturating_sub(5)..]);

    // Frame boundary analysis
    let frame_size = 1920;
    let num_frames = pcm.len() / frame_size;
    let mut jumps = Vec::new();

    for i in 1..num_frames {
        let boundary = i * frame_size;
        if boundary < pcm.len() {
            let before = pcm[boundary - 1];
            let after = pcm[boundary];
            let jump = (after - before).abs();
            if jump > 0.01 {
                jumps.push((i, jump));
            }
        }
    }

    println!(
        "  Frame jumps > 0.01: {} (of {} boundaries)",
        jumps.len(),
        num_frames - 1
    );
    if !jumps.is_empty() {
        let max_jump = jumps.iter().map(|(_, j)| *j).fold(0.0, f32::max);
        println!("  Max jump: {:.4}", max_jump);
    }
}
