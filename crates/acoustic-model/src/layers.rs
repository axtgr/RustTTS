//! Neural network layers for the acoustic model.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::AcousticModelConfig;

/// Trait for rotary position embeddings.
///
/// This trait unifies standard RoPE and multimodal RoPE (M-RoPE).
pub trait RotaryEmbeddingTrait {
    /// Apply rotary embedding to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - query tensor [batch, num_heads, seq, head_dim]
    /// * `k` - key tensor [batch, num_kv_heads, seq, head_dim]
    /// * `position_offset` - starting position for KV cache
    fn apply(&self, q: &Tensor, k: &Tensor, position_offset: usize) -> Result<(Tensor, Tensor)>;
}

/// RMSNorm layer for pre-normalization.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create a new RMSNorm layer.
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        Ok(Self { weight, eps })
    }

    /// Create RMSNorm with ones initialization (for testing).
    pub fn new_ones(hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    /// Apply RMSNorm to the input tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // Compute RMS: sqrt(mean(x^2))
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;

        // Normalize and scale
        let normalized = x.broadcast_div(&rms)?;
        let scaled = normalized.broadcast_mul(&self.weight)?;

        scaled.to_dtype(dtype)
    }
}

/// Multimodal Rotary Position Embedding (M-RoPE) for Qwen3-TTS.
///
/// This implements the interleaved multimodal RoPE used in Qwen3-TTS.
/// For text-only TTS, all 3 position components (temporal, height, width) are identical,
/// but different frequency patterns are applied to different parts of head_dim.
///
/// mrope_section = [24, 20, 20] means:
/// - First section (temporal): 24 dimensions
/// - Second section (height): 20 dimensions  
/// - Third section (width): 20 dimensions
/// Total = 64 = head_dim / 2
#[derive(Debug, Clone)]
pub struct MultimodalRotaryEmbedding {
    /// cos cache for each modality: shape [3, max_pos, head_dim]
    cos_cache: Tensor,
    /// sin cache for each modality: shape [3, max_pos, head_dim]
    sin_cache: Tensor,
    /// mrope section sizes [temporal, height, width]
    mrope_section: Vec<usize>,
    /// Whether to use interleaved pattern
    interleaved: bool,
    #[allow(dead_code)]
    dim: usize,
}

impl MultimodalRotaryEmbedding {
    /// Create a new MultimodalRotaryEmbedding.
    ///
    /// # Arguments
    /// * `dim` - head dimension (e.g., 128)
    /// * `max_position` - maximum sequence length
    /// * `theta` - RoPE theta (e.g., 1_000_000.0)
    /// * `mrope_section` - section sizes [temporal, height, width], sum should equal dim/2
    /// * `interleaved` - whether to use interleaved pattern
    /// * `device` - computation device
    pub fn new(
        dim: usize,
        max_position: usize,
        theta: f32,
        mrope_section: &[usize],
        interleaved: bool,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = dim / 2;

        // Validate mrope_section
        let section_sum: usize = mrope_section.iter().sum();
        if section_sum != half_dim {
            return Err(candle_core::Error::Msg(format!(
                "mrope_section sum ({}) must equal head_dim/2 ({})",
                section_sum, half_dim
            )));
        }

        // Compute inverse frequencies for full head_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq_tensor = Tensor::new(inv_freq.as_slice(), device)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_position).map(|i| i as f32).collect();
        let positions_tensor = Tensor::new(positions.as_slice(), device)?;

        // Compute freqs = positions * inv_freq: [max_pos, half_dim]
        let freqs = positions_tensor
            .unsqueeze(1)?
            .matmul(&inv_freq_tensor.unsqueeze(0)?)?;

        // Create cos/sin for each modality
        // For text-only TTS, all modalities use the same positions,
        // so we create 3 identical copies that will be combined differently
        let cos_single = freqs.cos()?;
        let sin_single = freqs.sin()?;

        // Stack into [3, max_pos, half_dim] - same values for all 3 modalities
        // (for text-only; for multimodal, different position_ids would be used)
        let cos_cache = Tensor::stack(&[&cos_single, &cos_single, &cos_single], 0)?;
        let sin_cache = Tensor::stack(&[&sin_single, &sin_single, &sin_single], 0)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            mrope_section: mrope_section.to_vec(),
            interleaved,
            dim,
        })
    }

    /// Apply multimodal rotary embedding to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - query tensor [batch, num_heads, seq, head_dim]
    /// * `k` - key tensor [batch, num_kv_heads, seq, head_dim]
    /// * `position_offset` - starting position for KV cache
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let head_dim = q.dim(3)?;
        let half_dim = head_dim / 2;

        // Get relevant slice: [3, seq_len, half_dim]
        let cos_slice = self.cos_cache.narrow(1, position_offset, seq_len)?;
        let sin_slice = self.sin_cache.narrow(1, position_offset, seq_len)?;

        // Combine modalities according to mrope pattern
        let (cos_combined, sin_combined) = if self.interleaved {
            self.combine_interleaved(&cos_slice, &sin_slice, half_dim)?
        } else {
            self.combine_concatenated(&cos_slice, &sin_slice)?
        };

        // cos_combined, sin_combined: [1, 1, seq_len, head_dim]
        // Duplicate half_dim to full head_dim: [cos, cos]
        let cos_full = Tensor::cat(&[&cos_combined, &cos_combined], 3)?;
        let sin_full = Tensor::cat(&[&sin_combined, &sin_combined], 3)?;

        // Apply rotary embedding
        let q_rot = apply_rotary_emb_mrope(q, &cos_full, &sin_full)?;
        let k_rot = apply_rotary_emb_mrope(k, &cos_full, &sin_full)?;

        Ok((q_rot, k_rot))
    }

    /// Combine modalities using interleaved pattern (Qwen3-TTS default).
    ///
    /// Pattern for mrope_section = [24, 20, 20]:
    /// - Start with modality 0 as base
    /// - For modality 1: fill indices 1, 4, 7, ... (step 3) with 20 values
    /// - For modality 2: fill indices 2, 5, 8, ... (step 3) with 20 values
    /// - Result: [m0, m1, m2, m0, m1, m2, ...] for first 60 indices, then m0 for 60-63
    fn combine_interleaved(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        half_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = cos.dim(1)?;
        let device = cos.device();

        // Get slices for each modality: [seq_len, half_dim]
        let cos0 = cos.i(0)?;
        let cos1 = cos.i(1)?;
        let cos2 = cos.i(2)?;
        let sin0 = sin.i(0)?;
        let sin1 = sin.i(1)?;
        let sin2 = sin.i(2)?;

        // Convert to vecs for manipulation
        let cos0_data: Vec<f32> = cos0.flatten_all()?.to_vec1()?;
        let cos1_data: Vec<f32> = cos1.flatten_all()?.to_vec1()?;
        let cos2_data: Vec<f32> = cos2.flatten_all()?.to_vec1()?;
        let sin0_data: Vec<f32> = sin0.flatten_all()?.to_vec1()?;
        let sin1_data: Vec<f32> = sin1.flatten_all()?.to_vec1()?;
        let sin2_data: Vec<f32> = sin2.flatten_all()?.to_vec1()?;

        let modality_num = 3;
        let mut cos_result = vec![0.0f32; seq_len * half_dim];
        let mut sin_result = vec![0.0f32; seq_len * half_dim];

        for s in 0..seq_len {
            let base = s * half_dim;

            // Start with modality 0
            for i in 0..half_dim {
                cos_result[base + i] = cos0_data[base + i];
                sin_result[base + i] = sin0_data[base + i];
            }

            // Apply modality 1 at indices 1, 4, 7, ... (step 3)
            // end_idx = mrope_section[1] * 3 = 20 * 3 = 60
            let end_idx_1 = self.mrope_section[1] * modality_num;
            for idx in (1..end_idx_1).step_by(modality_num) {
                if idx < half_dim {
                    cos_result[base + idx] = cos1_data[base + idx];
                    sin_result[base + idx] = sin1_data[base + idx];
                }
            }

            // Apply modality 2 at indices 2, 5, 8, ... (step 3)
            // end_idx = mrope_section[2] * 3 = 20 * 3 = 60
            let end_idx_2 = self.mrope_section[2] * modality_num;
            for idx in (2..end_idx_2).step_by(modality_num) {
                if idx < half_dim {
                    cos_result[base + idx] = cos2_data[base + idx];
                    sin_result[base + idx] = sin2_data[base + idx];
                }
            }
        }

        // Reshape to [1, 1, seq_len, half_dim]
        let cos_tensor = Tensor::new(cos_result.as_slice(), device)?
            .reshape((seq_len, half_dim))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin_tensor = Tensor::new(sin_result.as_slice(), device)?
            .reshape((seq_len, half_dim))?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        Ok((cos_tensor, sin_tensor))
    }

    /// Combine modalities using concatenated pattern (non-interleaved).
    ///
    /// Pattern for mrope_section = [16, 24, 24] (doubled to [16, 24, 24, 16, 24, 24]):
    /// - First 16 dims from modality 0
    /// - Next 24 dims from modality 1
    /// - Next 24 dims from modality 2
    /// - Repeat for second half
    fn combine_concatenated(&self, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = cos.dim(1)?;
        let half_dim = cos.dim(2)?;
        let device = cos.device();

        // Double the section: [s0, s1, s2, s0, s1, s2]
        let doubled_section: Vec<usize> = self
            .mrope_section
            .iter()
            .chain(self.mrope_section.iter())
            .copied()
            .collect();

        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();

        let mut offset = 0;
        for (i, &section_size) in doubled_section.iter().enumerate() {
            let modality = i % 3;
            let cos_mod = cos.i(modality)?;
            let sin_mod = sin.i(modality)?;

            // Take section_size elements starting from offset
            if offset + section_size <= half_dim * 2 {
                // Handle wrap-around for second half
                let actual_offset = offset % half_dim;
                let actual_size = section_size.min(half_dim - actual_offset);

                cos_parts.push(cos_mod.narrow(2, actual_offset, actual_size)?);
                sin_parts.push(sin_mod.narrow(2, actual_offset, actual_size)?);
            }

            offset += section_size;
        }

        // Concatenate all parts
        let cos_combined = Tensor::cat(&cos_parts, 2)?.unsqueeze(0)?;
        let sin_combined = Tensor::cat(&sin_parts, 2)?.unsqueeze(0)?;

        Ok((cos_combined, sin_combined))
    }
}

/// Apply rotary embedding using the standard rotate_half method.
fn apply_rotary_emb_mrope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    let half_dim = head_dim / 2;

    // Split x into two halves
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;

    // rotate_half: [-x2, x1]
    let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

    // q_embed = q * cos + rotate_half(q) * sin
    let result = x
        .broadcast_mul(cos)?
        .broadcast_add(&x_rotated.broadcast_mul(sin)?)?;

    Ok(result)
}

/// Standard Rotary Position Embedding (RoPE) for non-multimodal use.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    #[allow(dead_code)]
    dim: usize,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding.
    pub fn new(dim: usize, max_position: usize, theta: f32, device: &Device) -> Result<Self> {
        // Compute inverse frequencies
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
            .collect();

        let inv_freq_tensor = Tensor::new(inv_freq.as_slice(), device)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_position).map(|i| i as f32).collect();
        let positions_tensor = Tensor::new(positions.as_slice(), device)?;

        // Compute freqs = positions * inv_freq
        let freqs = positions_tensor
            .unsqueeze(1)?
            .matmul(&inv_freq_tensor.unsqueeze(0)?)?;

        // Compute cos and sin caches
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            cos_cache,
            sin_cache,
            dim,
        })
    }

    /// Apply rotary embedding to query and key tensors.
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(candle_core::D::Minus2)?;

        // Get the relevant slice of cos/sin caches
        let cos = self.cos_cache.narrow(0, position_offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, position_offset, seq_len)?;

        // Apply rotary embedding
        let q_rot = apply_rotary_emb(q, &cos, &sin)?;
        let k_rot = apply_rotary_emb(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

impl RotaryEmbeddingTrait for RotaryEmbedding {
    fn apply(&self, q: &Tensor, k: &Tensor, position_offset: usize) -> Result<(Tensor, Tensor)> {
        self.forward(q, k, position_offset)
    }
}

impl RotaryEmbeddingTrait for MultimodalRotaryEmbedding {
    fn apply(&self, q: &Tensor, k: &Tensor, position_offset: usize) -> Result<(Tensor, Tensor)> {
        self.forward(q, k, position_offset)
    }
}

/// Apply rotary embedding to a tensor (standard method).
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let half_dim = dims[dims.len() - 1] / 2;

    // Split x into two halves
    let x1 = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;

    // Reshape cos/sin for broadcasting
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, dim/2]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    // Concatenate
    Tensor::cat(&[rotated_x1, rotated_x2], candle_core::D::Minus1)
}

/// Multi-head attention layer with GQA support and optional QK-Norm.
#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Optional Q normalization (per head_dim).
    q_norm: Option<RmsNorm>,
    /// Optional K normalization (per head_dim).
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    /// Create a new attention layer.
    ///
    /// Qwen3-TTS uses QK-Norm (per-head RMSNorm on Q and K after projection).
    /// Attention projections have no bias (attention_bias: false in config).
    pub fn new(config: &AcousticModelConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim;

        // Q projection: hidden_size -> num_heads * head_dim (no bias)
        let q_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        // K projection: hidden_size -> num_kv_heads * head_dim (no bias)
        let k_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.num_kv_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        // V projection: hidden_size -> num_kv_heads * head_dim (no bias)
        let v_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.num_kv_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        // O projection: num_heads * head_dim -> hidden_size (no bias)
        let o_proj = candle_nn::linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        // Try to load QK-Norm weights (Qwen3-TTS specific)
        let q_norm = RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm")).ok();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    /// Forward pass with optional KV cache.
    pub fn forward<R: RotaryEmbeddingTrait>(
        &self,
        x: &Tensor,
        rope: &R,
        position_offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_with_debug(x, rope, position_offset, kv_cache, false)
    }

    /// Forward pass with optional debug output for layer 0.
    pub fn forward_with_debug<R: RotaryEmbeddingTrait>(
        &self,
        x: &Tensor,
        rope: &R,
        position_offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
        debug: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        if debug {
            let q_vals: Vec<f32> = q.i((0, seq_len - 1, ..10))?.to_vec1()?;
            tracing::debug!("Q proj last pos, first 10: {:?}", q_vals);
        }

        // Reshape for multi-head attention
        let mut q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, H, S, D]

        let mut k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-Norm if available (Qwen3-TTS specific)
        // QK-Norm is applied per-head on the last dimension (head_dim)
        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q)?;
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k)?;
        }

        if debug {
            let q_normed: Vec<f32> = q.i((0, 0, seq_len - 1, ..10))?.to_vec1()?;
            tracing::trace!("Q normed head0 last pos, first 10: {:?}", q_normed);
        }

        // Apply rotary embeddings
        let (q, k) = rope.apply(&q, &k, position_offset)?;

        if debug {
            let q_rope: Vec<f32> = q.i((0, 0, seq_len - 1, ..10))?.to_vec1()?;
            tracing::trace!("Q rope head0 last pos, first 10: {:?}", q_rope);
            let k_rope: Vec<f32> = k.i((0, 0, seq_len - 1, ..10))?.to_vec1()?;
            tracing::trace!("K rope head0 last pos, first 10: {:?}", k_rope);
        }

        // Handle KV cache
        let (k, v) = if let Some((cached_k, cached_v)) = kv_cache {
            let k = Tensor::cat(&[cached_k, &k], 2)?;
            let v = Tensor::cat(&[cached_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        // GQA: repeat KV heads if needed
        let k_expanded = repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v_expanded = repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        // Scaled dot-product attention
        let attn_weights = (q
            .matmul(&k_expanded.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?)?
            * self.scale as f64)?;

        if debug {
            // Attention weights before mask, head 0, last query position
            let attn_raw: Vec<f32> = attn_weights.i((0, 0, seq_len - 1, ..))?.to_vec1()?;
            tracing::debug!(
                "Attn weights (before mask) head0 last query, first 12: {:?}",
                &attn_raw[..12.min(attn_raw.len())]
            );
        }

        // Causal mask
        let total_seq_len = k_expanded.dim(2)?;
        let mask = create_causal_mask(seq_len, total_seq_len, x.device())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        if debug {
            // Attention weights after mask, head 0, last query position
            let attn_masked: Vec<f32> = attn_weights.i((0, 0, seq_len - 1, ..))?.to_vec1()?;
            tracing::debug!(
                "Attn weights (after mask) head0 last query, first 12: {:?}",
                &attn_masked[..12.min(attn_masked.len())]
            );
        }

        // Softmax and value projection
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        if debug {
            // Attention probs after softmax, head 0, last query position
            let attn_probs: Vec<f32> = attn_weights.i((0, 0, seq_len - 1, ..))?.to_vec1()?;
            tracing::debug!(
                "Attn probs (after softmax) head0 last query: {:?}",
                &attn_probs[..12.min(attn_probs.len())]
            );
        }

        let attn_output = attn_weights.matmul(&v_expanded)?;

        if debug {
            // Attention output before o_proj, head 0, last position
            let attn_out: Vec<f32> = attn_output.i((0, 0, seq_len - 1, ..10))?.to_vec1()?;
            tracing::debug!(
                "Attn output (before o_proj) head0 last pos, first 10: {:?}",
                attn_out
            );
        }

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        let output = self.o_proj.forward(&attn_output)?;

        if debug {
            // Attention output after o_proj, last position
            let o_proj_out: Vec<f32> = output.i((0, seq_len - 1, ..20))?.to_vec1()?;
            tracing::debug!(
                "Attn output (after o_proj) last pos, first 20: {:?}",
                o_proj_out
            );
        }

        // Return output and updated KV cache (original k, v without expansion)
        Ok((output, k, v))
    }
}

/// Repeat KV heads for GQA.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }

    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;

    Ok(x)
}

/// Create a causal attention mask.
fn create_causal_mask(query_len: usize, key_len: usize, device: &Device) -> Result<Tensor> {
    // Create lower triangular mask manually
    let mut mask_data = vec![0.0f32; query_len * key_len];
    for i in 0..query_len {
        for j in 0..key_len {
            // For causal mask: position i can attend to positions 0..=i+(key_len-query_len)
            let offset = key_len - query_len;
            if j <= i + offset {
                mask_data[i * key_len + j] = 0.0; // Can attend
            } else {
                mask_data[i * key_len + j] = f32::NEG_INFINITY; // Cannot attend
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (query_len, key_len), device)?;

    // Add batch and head dimensions
    mask.unsqueeze(0)?.unsqueeze(0)
}

/// MLP layer with SwiGLU activation.
#[derive(Debug)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    /// Create a new MLP layer.
    ///
    /// Qwen3-TTS MLP has no bias on any projections.
    pub fn new(config: &AcousticModelConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = candle_nn::linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Transformer decoder block.
#[derive(Debug)]
pub struct TransformerBlock {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: MLP,
}

impl TransformerBlock {
    /// Create a new transformer block.
    pub fn new(config: &AcousticModelConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let self_attn = Attention::new(config, vb.pp("self_attn"))?;
        let post_attention_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Forward pass with optional KV cache.
    pub fn forward<R: RotaryEmbeddingTrait>(
        &self,
        x: &Tensor,
        rope: &R,
        position_offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_with_debug(x, rope, position_offset, kv_cache, false)
    }

    /// Forward pass with optional debug output.
    pub fn forward_with_debug<R: RotaryEmbeddingTrait>(
        &self,
        x: &Tensor,
        rope: &R,
        position_offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
        debug: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len = x.dim(1)?;

        // Pre-norm + attention + residual
        let normed = self.input_layernorm.forward(x)?;

        if debug && position_offset == 0 && seq_len > 1 {
            let normed_vals: Vec<f32> = normed.i((0, seq_len - 1, ..20))?.to_vec1()?;
            tracing::trace!(
                "After input_layernorm, last pos, first 20: {:?}",
                normed_vals
            );
        }

        let (attn_output, k_cache, v_cache) =
            self.self_attn
                .forward_with_debug(&normed, rope, position_offset, kv_cache, debug)?;
        let x = (x + attn_output)?;

        if debug && position_offset == 0 && x.dim(1)? > 1 {
            let after_attn: Vec<f32> = x.i((0, x.dim(1)? - 1, ..20))?.to_vec1()?;
            tracing::trace!(
                "After attention + residual, last pos, first 20: {:?}",
                after_attn
            );
        }

        // Pre-norm + MLP + residual
        let normed = self.post_attention_layernorm.forward(&x)?;

        if debug && position_offset == 0 && normed.dim(1)? > 1 {
            let post_ln: Vec<f32> = normed.i((0, normed.dim(1)? - 1, ..20))?.to_vec1()?;
            tracing::trace!(
                "After post_attention_layernorm, last pos, first 20: {:?}",
                post_ln
            );
        }

        let mlp_output = self.mlp.forward(&normed)?;

        if debug && position_offset == 0 && mlp_output.dim(1)? > 1 {
            let mlp_out: Vec<f32> = mlp_output.i((0, mlp_output.dim(1)? - 1, ..20))?.to_vec1()?;
            tracing::trace!("MLP output, last pos, first 20: {:?}", mlp_out);
        }

        let x = (x + mlp_output)?;

        if debug && position_offset == 0 && x.dim(1)? > 1 {
            let final_out: Vec<f32> = x.i((0, x.dim(1)? - 1, ..20))?.to_vec1()?;
            tracing::debug!(
                "Layer output (after MLP + residual), last pos, first 20: {:?}",
                final_out
            );
        }

        Ok((x, k_cache, v_cache))
    }
}

/// Text projection layer for Qwen3-TTS.
///
/// Projects text embeddings from embedding_dim (2048) to hidden_size (1024).
/// Architecture: Linear(2048, 2048) -> SiLU -> Linear(2048, 1024)
#[derive(Debug)]
pub struct TextProjection {
    fc1: Linear,
    fc2: Linear,
}

impl TextProjection {
    /// Create a new text projection layer.
    ///
    /// TextProjection has bias unlike attention/MLP layers.
    pub fn new(embedding_dim: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(embedding_dim, embedding_dim, vb.pp("linear_fc1"))?;
        let fc2 = candle_nn::linear(embedding_dim, hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    /// Forward pass through the projection.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = candle_nn::ops::silu(&x)?;
        self.fc2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let device = Device::Cpu;
        let norm = RmsNorm::new_ones(64, 1e-6, &device).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (2, 10, 64), &device).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.dims(), x.dims());
    }

    #[test]
    fn test_rotary_embedding() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 1024, 10000.0, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device).unwrap();

        let (q_rot, k_rot) = rope.forward(&q, &k, 0).unwrap();

        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(3, 5, &device).unwrap();

        assert_eq!(mask.dims(), &[1, 1, 3, 5]);
    }
}
