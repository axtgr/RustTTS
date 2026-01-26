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
