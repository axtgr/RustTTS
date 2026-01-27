//! KV cache implementation for efficient autoregressive generation.

use candle_core::{D, DType, Device, Result, Tensor};

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
    pub cached_mask: Option<(usize, usize, Tensor)>,
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
            cached_mask: None,
        })
    }

    /// Append new K, V tokens to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        // k, v shape: [batch, num_kv_heads, seq_len, head_dim]
        let combined_k = Tensor::cat(&[&self.active_k, k], 2)?;
        let combined_v = Tensor::cat(&[&self.active_v, v], 2)?;
        let total_len = combined_k.dim(2)?;

        let mut offset = 0;
        while total_len - offset >= self.chunk_size {
            let k_chunk = combined_k.narrow(D::Minus2, offset, self.chunk_size)?;
            let v_chunk = combined_v.narrow(D::Minus2, offset, self.chunk_size)?;
            self.frozen.push((k_chunk, v_chunk));
            offset += self.chunk_size;
        }

        let remaining = total_len - offset;
        let (b, h, _, d) = combined_k.dims4()?;
        let device = combined_k.device().clone();
        let dtype = combined_k.dtype();

        if remaining > 0 {
            self.active_k = combined_k.narrow(D::Minus2, offset, remaining)?;
            self.active_v = combined_v.narrow(D::Minus2, offset, remaining)?;
            self.active_len = remaining;
        } else {
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
