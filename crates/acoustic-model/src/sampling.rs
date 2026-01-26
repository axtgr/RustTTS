//! Sampling strategies for token generation.

use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for scaling logits.
    pub temperature: f32,
    /// Top-k parameter (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) parameter (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled, >1.0 = discourage repetition).
    pub repetition_penalty: f32,
    /// Random seed (None = random).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create greedy sampling configuration (argmax, no randomness).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0, // Will trigger argmax
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(0),
        }
    }

    /// Create sampling with specific temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Create top-k sampling configuration.
    pub fn top_k(k: usize) -> Self {
        Self {
            top_k: k,
            ..Default::default()
        }
    }

    /// Create top-p (nucleus) sampling configuration.
    pub fn top_p(p: f32) -> Self {
        Self {
            top_p: p,
            ..Default::default()
        }
    }
}

/// Token sampler for autoregressive generation.
#[derive(Debug)]
pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

/// Apply repetition penalty to logits for previously generated tokens.
///
/// Following HuggingFace transformers implementation:
/// - If logit < 0: multiply by penalty (makes more negative)
/// - If logit >= 0: divide by penalty (makes less positive)
///
/// This discourages the model from repeating tokens it has already generated.
pub fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        // No penalty to apply
        return;
    }

    for &token in generated_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] < 0.0 {
                logits[idx] *= penalty;
            } else {
                logits[idx] /= penalty;
            }
        }
    }
}

impl Sampler {
    /// Create a new sampler with the given configuration.
    pub fn new(config: SamplingConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Sample a token from logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Greedy decoding: temperature=0 or very low means argmax
        if self.config.temperature < f32::EPSILON {
            return self.greedy(logits);
        }

        // Apply temperature
        let scaled: Vec<f32> = if (self.config.temperature - 1.0).abs() > f32::EPSILON {
            logits
                .iter()
                .map(|&x| x / self.config.temperature)
                .collect()
        } else {
            logits.to_vec()
        };

        // Convert to probabilities via softmax
        let mut probs = softmax(&scaled);

        // Apply top-k filtering
        if self.config.top_k > 0 && self.config.top_k < probs.len() {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let threshold = indexed[self.config.top_k - 1].1;
            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                }
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0;
            let mut cutoff_idx = indexed.len();
            for (i, (_, p)) in indexed.iter().enumerate() {
                cumsum += p;
                if cumsum > self.config.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            let threshold = if cutoff_idx < indexed.len() {
                indexed[cutoff_idx].1
            } else {
                0.0
            };

            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                }
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Sample from the distribution
        if let Ok(dist) = WeightedIndex::new(&probs) {
            dist.sample(&mut self.rng) as u32
        } else {
            // Fallback to argmax
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        }
    }

    /// Greedy decoding (argmax).
    pub fn greedy(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

/// Compute softmax of a slice of values.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    if sum > 0.0 {
        exp.iter().map(|&x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let config = SamplingConfig::default();
        let sampler = Sampler::new(config);

        let logits = vec![0.1, 0.5, 0.3, 0.1];
        assert_eq!(sampler.greedy(&logits), 1);

        let logits = vec![0.9, 0.1, 0.0, 0.0];
        assert_eq!(sampler.greedy(&logits), 0);
    }

    #[test]
    fn test_deterministic_sampling() {
        let config = SamplingConfig {
            seed: Some(42),
            ..Default::default()
        };

        let logits = vec![0.25, 0.25, 0.25, 0.25];

        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);

        let result1 = sampler1.sample(&logits);
        let result2 = sampler2.sample(&logits);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check ordering
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_temperature_scaling() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        // Low temperature should make distribution more peaked
        let config_low = SamplingConfig {
            temperature: 0.5,
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler_low = Sampler::new(config_low);

        // High temperature should make distribution more uniform
        let config_high = SamplingConfig {
            temperature: 2.0,
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler_high = Sampler::new(config_high);

        // Both should work without panicking
        let _ = sampler_low.sample(&logits);
        let _ = sampler_high.sample(&logits);
    }
}
