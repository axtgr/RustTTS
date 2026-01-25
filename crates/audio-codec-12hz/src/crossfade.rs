//! Overlap-add with Hann window crossfade for seamless audio chunks.

use std::f32::consts::PI;

/// Crossfader for seamless audio chunk concatenation.
#[derive(Debug)]
pub struct Crossfader {
    /// Crossfade duration in samples.
    fade_samples: usize,
    /// Hann window for fade-in.
    fade_in_window: Vec<f32>,
    /// Hann window for fade-out.
    fade_out_window: Vec<f32>,
    /// Buffer for overlap region.
    overlap_buffer: Vec<f32>,
}

impl Crossfader {
    /// Create a new crossfader with the specified fade duration.
    ///
    /// # Arguments
    /// * `fade_ms` - Crossfade duration in milliseconds
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(fade_ms: f32, sample_rate: u32) -> Self {
        let fade_samples = ((fade_ms / 1000.0) * sample_rate as f32) as usize;
        let fade_samples = fade_samples.max(1);

        // Generate Hann windows
        let fade_in_window: Vec<f32> = (0..fade_samples)
            .map(|i| {
                let t = i as f32 / (fade_samples - 1).max(1) as f32;
                0.5 * (1.0 - (PI * t).cos())
            })
            .collect();

        let fade_out_window: Vec<f32> = fade_in_window.iter().rev().copied().collect();

        Self {
            fade_samples,
            fade_in_window,
            fade_out_window,
            overlap_buffer: Vec::new(),
        }
    }

    /// Get the crossfade duration in samples.
    pub fn fade_samples(&self) -> usize {
        self.fade_samples
    }

    /// Process a chunk of audio, applying crossfade with the previous chunk.
    ///
    /// Returns the processed audio samples.
    pub fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        if chunk.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(chunk.len());

        // If we have overlap from previous chunk, apply crossfade
        if !self.overlap_buffer.is_empty() {
            let overlap_len = self
                .overlap_buffer
                .len()
                .min(chunk.len())
                .min(self.fade_samples);

            // Apply crossfade in overlap region
            for (i, (&prev, &curr)) in self.overlap_buffer[..overlap_len]
                .iter()
                .zip(&chunk[..overlap_len])
                .enumerate()
            {
                let prev_sample = prev * self.fade_out_window[i];
                let curr_sample = curr * self.fade_in_window[i];
                output.push(prev_sample + curr_sample);
            }

            // Add remaining samples from current chunk (if overlap was shorter)
            if overlap_len < chunk.len() {
                output.extend_from_slice(
                    &chunk[overlap_len..chunk.len().saturating_sub(self.fade_samples)],
                );
            }
        } else {
            // No previous overlap, just copy (minus fade region at end)
            let main_len = chunk.len().saturating_sub(self.fade_samples);
            output.extend_from_slice(&chunk[..main_len]);
        }

        // Store the tail for next crossfade
        self.overlap_buffer.clear();
        if chunk.len() >= self.fade_samples {
            self.overlap_buffer
                .extend_from_slice(&chunk[chunk.len() - self.fade_samples..]);
        } else {
            self.overlap_buffer.extend_from_slice(chunk);
        }

        output
    }

    /// Flush any remaining samples in the buffer.
    pub fn flush(&mut self) -> Vec<f32> {
        let output: Vec<f32> = self
            .overlap_buffer
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                if i < self.fade_out_window.len() {
                    sample * self.fade_out_window[i]
                } else {
                    sample
                }
            })
            .collect();

        self.overlap_buffer.clear();
        output
    }

    /// Reset the crossfader state.
    pub fn reset(&mut self) {
        self.overlap_buffer.clear();
    }
}

/// Apply Hann window to audio samples.
pub fn apply_hann_window(samples: &mut [f32]) {
    let n = samples.len();
    if n == 0 {
        return;
    }

    for (i, sample) in samples.iter_mut().enumerate() {
        let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1).max(1) as f32).cos());
        *sample *= window;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossfader_creation() {
        let crossfader = Crossfader::new(10.0, 24000);
        // 10ms at 24kHz = 240 samples
        assert_eq!(crossfader.fade_samples(), 240);
    }

    #[test]
    fn test_hann_window_symmetry() {
        let crossfader = Crossfader::new(10.0, 24000);

        // Fade-in and fade-out should be symmetric
        let n = crossfader.fade_in_window.len();
        for i in 0..n {
            let diff = (crossfader.fade_in_window[i] - crossfader.fade_out_window[n - 1 - i]).abs();
            assert!(diff < 1e-6, "Windows should be symmetric");
        }
    }

    #[test]
    fn test_crossfade_sum_to_one() {
        let crossfader = Crossfader::new(10.0, 24000);

        // At any point in the crossfade region, fade_in + fade_out should ≈ 1
        for i in 0..crossfader.fade_samples() {
            let sum = crossfader.fade_in_window[i] + crossfader.fade_out_window[i];
            assert!((sum - 1.0).abs() < 1e-6, "Crossfade should sum to 1.0");
        }
    }

    #[test]
    fn test_process_single_chunk() {
        let mut crossfader = Crossfader::new(5.0, 24000);
        let chunk = vec![1.0f32; 1000];

        let output = crossfader.process(&chunk);
        // First chunk should output all but the fade region
        assert!(output.len() < chunk.len());
    }

    #[test]
    fn test_flush() {
        let mut crossfader = Crossfader::new(5.0, 24000);
        let chunk = vec![1.0f32; 1000];

        crossfader.process(&chunk);
        let flushed = crossfader.flush();

        // Should output the remaining samples with fade-out
        assert!(!flushed.is_empty());
    }

    #[test]
    fn test_apply_hann_window() {
        let mut samples = vec![1.0f32; 100];
        apply_hann_window(&mut samples);

        // First and last samples should be near zero
        assert!(samples[0].abs() < 0.01);
        assert!(samples[99].abs() < 0.01);

        // Middle should be near 1.0
        assert!((samples[50] - 1.0).abs() < 0.1);
    }
}
