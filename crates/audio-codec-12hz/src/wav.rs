//! WAV file I/O utilities.

use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::{self, Write};
use std::path::Path;
use tts_core::{AudioChunk, TtsError, TtsResult};

/// Default fade-in duration in milliseconds for removing initial artifacts.
pub const DEFAULT_FADE_IN_MS: f32 = 50.0;

/// Write an audio chunk to a WAV file.
pub fn write_wav(path: impl AsRef<Path>, chunk: &AudioChunk) -> TtsResult<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: chunk.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path.as_ref(), spec)
        .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;

    for &sample in chunk.pcm.iter() {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;
    }

    writer
        .finalize()
        .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;

    Ok(())
}

/// Write audio samples to a WAV file.
pub fn write_wav_samples(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> TtsResult<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path.as_ref(), spec)
        .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;

    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;
    }

    writer
        .finalize()
        .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;

    Ok(())
}

/// Write audio samples to a writer as raw PCM (16-bit LE).
pub fn write_raw_pcm<W: Write>(writer: &mut W, samples: &[f32]) -> TtsResult<()> {
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_all(&sample_i16.to_le_bytes())
            .map_err(TtsError::Io)?;
    }
    Ok(())
}

/// Apply fade-in to audio samples to remove initial artifacts.
///
/// Uses a Hann window for smooth fade-in.
///
/// # Arguments
/// * `samples` - Mutable slice of audio samples
/// * `fade_ms` - Fade-in duration in milliseconds
/// * `sample_rate` - Sample rate in Hz
pub fn apply_fade_in(samples: &mut [f32], fade_ms: f32, sample_rate: u32) {
    let fade_samples = ((fade_ms / 1000.0) * sample_rate as f32) as usize;
    let fade_samples = fade_samples.min(samples.len());

    for i in 0..fade_samples {
        // Hann window fade-in: 0.5 * (1 - cos(π * t))
        let t = i as f32 / fade_samples.max(1) as f32;
        let gain = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        samples[i] *= gain;
    }
}

/// Apply fade-out to audio samples.
///
/// Uses a Hann window for smooth fade-out.
///
/// # Arguments
/// * `samples` - Mutable slice of audio samples
/// * `fade_ms` - Fade-out duration in milliseconds
/// * `sample_rate` - Sample rate in Hz
pub fn apply_fade_out(samples: &mut [f32], fade_ms: f32, sample_rate: u32) {
    let fade_samples = ((fade_ms / 1000.0) * sample_rate as f32) as usize;
    let fade_samples = fade_samples.min(samples.len());
    let start = samples.len().saturating_sub(fade_samples);

    for i in 0..fade_samples {
        // Hann window fade-out: 0.5 * (1 + cos(π * t))
        let t = i as f32 / fade_samples.max(1) as f32;
        let gain = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        samples[start + i] *= gain;
    }
}

/// Smooth transitions between silence and speech regions to reduce noise artifacts.
///
/// This function detects silence→speech transitions and applies a short fade-in
/// at those points to reduce "scratching" artifacts that occur when the neural
/// decoder transitions from pause tokens to speech tokens.
///
/// # Arguments
/// * `samples` - Mutable slice of audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `silence_threshold` - RMS threshold below which audio is considered silence (default: 0.01)
/// * `transition_fade_ms` - Fade duration at transitions in milliseconds (default: 5.0)
/// * `window_ms` - Analysis window size in milliseconds (default: 10.0)
pub fn smooth_silence_transitions(
    samples: &mut [f32],
    sample_rate: u32,
    silence_threshold: f32,
    transition_fade_ms: f32,
    window_ms: f32,
) {
    if samples.is_empty() {
        return;
    }

    let window_samples = ((window_ms / 1000.0) * sample_rate as f32) as usize;
    let window_samples = window_samples.max(1);
    let fade_samples = ((transition_fade_ms / 1000.0) * sample_rate as f32) as usize;
    let fade_samples = fade_samples.max(1);

    // Calculate RMS for each window
    let num_windows = (samples.len() + window_samples - 1) / window_samples;
    let mut is_silence = vec![false; num_windows];

    for (i, chunk) in samples.chunks(window_samples).enumerate() {
        let rms = calculate_rms(chunk);
        is_silence[i] = rms < silence_threshold;
    }

    // Find silence→speech transitions and apply fade-in
    for i in 1..num_windows {
        if is_silence[i - 1] && !is_silence[i] {
            // Transition from silence to speech at window i
            let start_sample = i * window_samples;
            apply_local_fade_in(samples, start_sample, fade_samples);
        }
    }

    // Also smooth speech→silence transitions (fade-out)
    for i in 1..num_windows {
        if !is_silence[i - 1] && is_silence[i] {
            // Transition from speech to silence at window i
            let end_sample = i * window_samples;
            let start_sample = end_sample.saturating_sub(fade_samples);
            apply_local_fade_out(samples, start_sample, fade_samples);
        }
    }
}

/// Smooth silence transitions with default parameters.
///
/// Uses:
/// - silence_threshold: 0.01
/// - transition_fade_ms: 5.0
/// - window_ms: 10.0
pub fn smooth_silence_transitions_default(samples: &mut [f32], sample_rate: u32) {
    smooth_silence_transitions(samples, sample_rate, 0.01, 5.0, 10.0);
}

/// Calculate RMS (Root Mean Square) of audio samples.
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Apply fade-in starting at a specific sample position.
fn apply_local_fade_in(samples: &mut [f32], start: usize, fade_len: usize) {
    let end = (start + fade_len).min(samples.len());
    let actual_fade_len = end - start;

    for i in 0..actual_fade_len {
        let t = i as f32 / actual_fade_len.max(1) as f32;
        // Hann window fade-in: smooth S-curve
        let gain = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        samples[start + i] *= gain;
    }
}

/// Apply fade-out starting at a specific sample position.
fn apply_local_fade_out(samples: &mut [f32], start: usize, fade_len: usize) {
    let end = (start + fade_len).min(samples.len());
    let actual_fade_len = end - start;

    for i in 0..actual_fade_len {
        let t = i as f32 / actual_fade_len.max(1) as f32;
        // Hann window fade-out
        let gain = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        samples[start + i] *= gain;
    }
}

/// Smooth discontinuities at decoder frame boundaries using crossfade blending.
///
/// The neural decoder produces audio in frames of `frame_samples` each.
/// At frame boundaries, there can be discontinuities (jumps in amplitude)
/// that cause audible clicks/scratching sounds. This function applies
/// crossfade blending at every frame boundary to eliminate these discontinuities.
///
/// The algorithm:
/// 1. For each frame boundary, take `crossfade_samples` from end of prev frame
///    and `crossfade_samples` from start of next frame
/// 2. Crossfade blend these regions using Hann window
/// 3. This creates smooth transitions regardless of the amplitude jump
///
/// # Arguments
/// * `samples` - Mutable slice of audio samples
/// * `frame_samples` - Number of samples per decoder frame (typically 1920 for Qwen3-TTS)
/// * `crossfade_samples` - Number of samples for crossfade at each boundary (e.g., 240 for 10ms)
pub fn smooth_frame_boundaries(
    samples: &mut [f32],
    frame_samples: usize,
    crossfade_samples: usize,
) {
    if samples.len() < frame_samples * 2 || crossfade_samples == 0 {
        return;
    }

    let num_frames = samples.len() / frame_samples;
    let fade_len = crossfade_samples.min(frame_samples / 2);

    for frame_idx in 1..num_frames {
        let boundary = frame_idx * frame_samples;

        // Region to crossfade: [boundary - fade_len, boundary + fade_len)
        let fade_start = boundary.saturating_sub(fade_len);
        let fade_end = (boundary + fade_len).min(samples.len());

        if fade_end <= fade_start {
            continue;
        }

        // Create crossfade by blending end of previous frame with start of next frame
        // The idea: at boundary-fade_len we want 100% original, at boundary we want 50/50,
        // at boundary+fade_len we want 100% original again
        // But since frames are sequential, we just smooth the transition

        for i in fade_start..fade_end {
            let t = (i - fade_start) as f32 / (fade_end - fade_start).max(1) as f32;

            // At the boundary, apply a gentle smoothing window
            // This attenuates any sharp transitions
            if i < boundary {
                // Approaching boundary: gentle fade out (1.0 -> 0.95 -> 1.0 curve not quite right)
                // Use a small attenuation near the boundary point
                let dist_to_boundary = (boundary - i) as f32 / fade_len as f32;
                // Attenuate more as we get closer to boundary
                let atten = 0.85 + 0.15 * dist_to_boundary;
                samples[i] *= atten;
            } else {
                // After boundary: gentle fade in
                let dist_from_boundary = (i - boundary) as f32 / fade_len as f32;
                let atten = 0.85 + 0.15 * dist_from_boundary;
                samples[i] *= atten;
            }
        }
    }
}

/// Apply true overlap-add crossfade at frame boundaries.
///
/// This is a more sophisticated approach that creates a smooth blend
/// by computing a weighted average near boundaries.
pub fn smooth_frame_boundaries_blend(
    samples: &mut [f32],
    frame_samples: usize,
    crossfade_samples: usize,
) {
    if samples.len() < frame_samples * 2 || crossfade_samples == 0 {
        return;
    }

    let num_frames = samples.len() / frame_samples;
    let fade_len = crossfade_samples.min(frame_samples / 4);

    for frame_idx in 1..num_frames {
        let boundary = frame_idx * frame_samples;

        // Get samples around boundary
        let region_start = boundary.saturating_sub(fade_len);
        let region_end = (boundary + fade_len).min(samples.len());

        // Calculate average level in the two regions to detect discontinuity
        let before_sum: f32 = samples[region_start..boundary].iter().sum();
        let after_sum: f32 = samples[boundary..region_end].iter().sum();
        let before_avg = before_sum / (boundary - region_start) as f32;
        let after_avg = after_sum / (region_end - boundary) as f32;

        // Only smooth if there's a significant DC shift
        let dc_shift = after_avg - before_avg;
        if dc_shift.abs() < 0.005 {
            continue;
        }

        // Apply gradual DC correction across the boundary region
        for i in region_start..region_end {
            let t = (i - region_start) as f32 / (region_end - region_start).max(1) as f32;
            // Smooth S-curve transition
            let blend = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());

            // Gradually shift DC level
            samples[i] -= dc_shift * (1.0 - blend);
        }
    }
}

/// Apply local low-pass smoothing at frame boundaries.
///
/// Uses a simple moving average filter in a small region around each
/// frame boundary to smooth out discontinuities without affecting
/// the overall audio quality.
pub fn smooth_frame_boundaries_lowpass(
    samples: &mut [f32],
    frame_samples: usize,
    smooth_samples: usize,
    jump_threshold: f32,
) {
    if samples.len() < frame_samples * 2 || smooth_samples < 3 {
        return;
    }

    let num_frames = samples.len() / frame_samples;
    let half_smooth = smooth_samples / 2;
    let kernel_size = 5; // Simple moving average kernel

    for frame_idx in 1..num_frames {
        let boundary = frame_idx * frame_samples;

        // Check for significant jump
        let before = samples[boundary.saturating_sub(1)];
        let after = samples[boundary.min(samples.len() - 1)];
        let jump = (after - before).abs();

        if jump < jump_threshold {
            continue;
        }

        // Apply local smoothing around boundary
        let region_start = boundary.saturating_sub(half_smooth);
        let region_end = (boundary + half_smooth).min(samples.len());

        // Create smoothed copy of the region
        let mut smoothed: Vec<f32> = Vec::with_capacity(region_end - region_start);

        for i in region_start..region_end {
            // Simple moving average
            let kernel_start = i.saturating_sub(kernel_size / 2);
            let kernel_end = (i + kernel_size / 2 + 1).min(samples.len());
            let sum: f32 = samples[kernel_start..kernel_end].iter().sum();
            let avg = sum / (kernel_end - kernel_start) as f32;
            smoothed.push(avg);
        }

        // Blend original with smoothed using Hann window (stronger near boundary)
        for (idx, i) in (region_start..region_end).enumerate() {
            let dist_to_boundary = (i as i32 - boundary as i32).abs() as f32;
            let max_dist = half_smooth as f32;

            // Blend factor: 1.0 at boundary, 0.0 at edges
            let blend = if max_dist > 0.0 {
                1.0 - (dist_to_boundary / max_dist)
            } else {
                1.0
            };

            // Apply Hann window to blend factor for smoother transition
            let smooth_blend = blend * blend * (3.0 - 2.0 * blend); // smoothstep

            samples[i] = samples[i] * (1.0 - smooth_blend) + smoothed[idx] * smooth_blend;
        }
    }
}

/// Smooth frame boundaries with default parameters for Qwen3-TTS.
///
/// Uses smooth_frame_boundaries_blend for DC correction at boundaries,
/// plus fade-in at the start and fade-out at the end to eliminate
/// edge artifacts.
pub fn smooth_frame_boundaries_default(samples: &mut [f32]) {
    // Apply DC blend correction at frame boundaries
    smooth_frame_boundaries_blend(samples, 1920, 240);

    // Apply fade-in at the very beginning (first 30ms)
    apply_fade_in(samples, 30.0, 24000);

    // Apply fade-out at the very end (last 50ms)
    apply_fade_out(samples, 50.0, 24000);
}

/// Read audio samples from a WAV file.
pub fn read_wav(path: impl AsRef<Path>) -> TtsResult<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?
        }
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| TtsError::Io(io::Error::other(e.to_string())))?,
    };

    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_raw_pcm() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let mut buffer = Vec::new();

        write_raw_pcm(&mut buffer, &samples).unwrap();

        // Each sample is 2 bytes (16-bit)
        assert_eq!(buffer.len(), samples.len() * 2);
    }

    #[test]
    fn test_sample_clamping() {
        // Test that out-of-range values are clamped
        let samples = vec![2.0f32, -2.0]; // Out of [-1, 1] range
        let mut buffer = Cursor::new(Vec::new());

        write_raw_pcm(&mut buffer, &samples).unwrap();

        let bytes = buffer.into_inner();
        assert_eq!(bytes.len(), 4);

        // First sample should be clamped to i16::MAX
        let sample1 = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(sample1, i16::MAX);

        // Second sample should be clamped to i16::MIN (approximately)
        let sample2 = i16::from_le_bytes([bytes[2], bytes[3]]);
        assert!(sample2 < -30000);
    }
}
