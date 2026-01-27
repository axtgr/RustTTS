//! KV cache implementation for efficient autoregressive generation.

use candle_core::{DType, Device, Result, Tensor};

/// A chunked KV cache for a single layer.
/// This avoids O(L^2) memory copying by keeping previous tokens in frozen chunks.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    /// Frozen chunks of (K, V) pairs.
    pub frozen: Vec<(Tensor, Tensor)>,
    /// Current active chunk being built.
    pub active_k: Tensor,
    pub active_v: Tensor,
    /// Maximum size of a chunk.
    pub chunk_size: usize,
    /// Current length of the active chunk.
    pub active_len: usize,
}

impl LayerKvCache {
    pub fn new(
        chunk_size: usize,
        head_dim: usize,
        num_kv_heads: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Initialize empty active chunks
        let active_k = Tensor::zeros((1, num_kv_heads, 0, head_dim), dtype, device)?;
        let active_v = Tensor::zeros((1, num_kv_heads, 0, head_dim), dtype, device)?;

        Ok(Self {
            frozen: Vec::new(),
            active_k,
            active_v,
            chunk_size,
            active_len: 0,
        })
    }

    /// Append new K, V tokens to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        // k, v shape: [batch, num_kv_heads, seq_len, head_dim]
        // Usually seq_len is 1 during generation.

        // Naive implementation for active chunk: cat
        // Since active chunk is small (< chunk_size), cat is cheap.
        self.active_k = Tensor::cat(&[&self.active_k, k], 2)?;
        self.active_v = Tensor::cat(&[&self.active_v, v], 2)?;
        self.active_len += k.dim(2)?;

        // If active chunk is full, freeze it
        if self.active_len >= self.chunk_size {
            // If we overshoot, we should split. But for 1-token generation it's exact.
            // If we receive multiple tokens (prefill), we might need to split.
            // For simplicity, let's assume we just freeze the whole thing if it exceeds.
            // Or better: keep it growing until next append?
            // Let's strictly freeze when >= chunk_size.

            self.frozen
                .push((self.active_k.clone(), self.active_v.clone()));

            // Reset active
            let (b, h, _, d) = self.active_k.dims4()?;
            let device = self.active_k.device().clone();
            let dtype = self.active_k.dtype();
            self.active_k = Tensor::zeros((b, h, 0, d), dtype, &device)?;
            self.active_v = Tensor::zeros((b, h, 0, d), dtype, &device)?;
            self.active_len = 0;
        }

        Ok(())
    }

    /// Get total sequence length stored in cache
    pub fn seq_len(&self) -> usize {
        let frozen_len: usize = self.frozen.iter().map(|(k, _)| k.dim(2).unwrap_or(0)).sum();
        frozen_len + self.active_len
    }
}
