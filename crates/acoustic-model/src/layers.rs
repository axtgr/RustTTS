//! Neural network layers for the acoustic model.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::AcousticModelConfig;

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

/// Rotary Position Embedding (RoPE).
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

/// Apply rotary embedding to a tensor.
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
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        position_offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

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

        // Apply rotary embeddings
        let (q, k) = rope.forward(&q, &k, position_offset)?;

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

        // Causal mask
        let total_seq_len = k_expanded.dim(2)?;
        let mask = create_causal_mask(seq_len, total_seq_len, x.device())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Softmax and value projection
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        let output = self.o_proj.forward(&attn_output)?;

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
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        position_offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Pre-norm + attention + residual
        let normed = self.input_layernorm.forward(x)?;
        let (attn_output, k_cache, v_cache) =
            self.self_attn
                .forward(&normed, rope, position_offset, kv_cache)?;
        let x = (x + attn_output)?;

        // Pre-norm + MLP + residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let x = (x + mlp_output)?;

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
