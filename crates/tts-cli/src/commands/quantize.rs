
//! Quantization command implementation.


use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, Context};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{Device, DType};
use tracing::info;

/// Options for quantization command.
#[derive(clap::Args)]
pub struct QuantizeOptions {
    /// Input model text/dir (expects structure with model.safetensors or direct path).
    #[arg(long, short = 'i')]
    pub input: PathBuf,

    /// Output GGUF file path.
    #[arg(long, short = 'o')]
    pub output: PathBuf,

    /// Quantization type (currently only q4_0 supported).
    #[arg(long, default_value = "q4_0")]
    pub qtype: String,
}

pub fn run(options: QuantizeOptions) -> Result<()> {
    let start = Instant::now();
    info!("Starting quantization of {:?}", options.input);

    // Determine input path (directory or file)
    let input_path = if options.input.is_dir() {
        options.input.join("model.safetensors")
    } else {
        options.input.clone()
    };

    if !input_path.exists() {
        anyhow::bail!("Input file not found: {:?}", input_path);
    }

    // Load safetensors - map to CPU
    let device = Device::Cpu;
    let tensors = candle_core::safetensors::load(&input_path, &device)
        .context("Failed to load safetensors")?;

    info!("Loaded {} tensors", tensors.len());

    let mut out_tensors: Vec<(String, QTensor)> = Vec::with_capacity(tensors.len());

    // Statistics
    let mut num_quantized = 0;
    let mut num_skipped = 0;

    for (name, tensor) in tensors {
        let shape = tensor.dims();
        let numel = tensor.elem_count();
        
        // rudimentary heuristic for quantization
        // Quantize 2D weights that are not norms or small embeddings.
        // Qwen3-TTS: talker.model.text_embedding, codec_embedding, layers...
        // We usually want to quantize heavy checking matrices.
        
        let should_quantize = if options.qtype == "q4_0" {
            // Must be at least 2D
            if shape.len() >= 2 {
                 // Skip Norms (usually 1D but just in case, check name)
                if name.contains("norm") {
                    false
                } else if name.contains("small_to_mtp_projection") {
                    // Critical projection layer for CodePredictor - keep high precision
                    false
                } else if name.contains("codec_embedding") {
                    // Codec embeddings are sensitive - keep high precision for now
                    false
                } else if name.contains("embedding") || name.contains("embed_tokens") {
                    // Other embeddings (text) usually safe to quantize
                    true 
                } else if name.ends_with(".weight") {
                    // Linear layers
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if should_quantize {
            // Convert to Q4_0
            // QTensor::quantize requires &Tensor, quantization method
            let q_tensor = QTensor::quantize(&tensor, candle_core::quantized::GgmlDType::Q4_0)?;
            out_tensors.push((name.clone(), q_tensor));
            num_quantized += 1;
            // info!("Quantized {} {:?}", name, shape);
        } else {
            // Keep as F32 (GGUF supports F32)
            // Or F16? `candle` QTensor supports F32/F16 variants.
            // But we need to wrap it into QTensor.
            // QTensor has a variant usually for F32?
            // Wait, QTensor::quantize with F32 type?
            
            // Actually, `gguf_file` expects `QTensor`.
            // QTensor::quantize with F32 will just store it as F32/F16?
            // Let's check GgmlDType.
            
            let target_type = if tensor.dtype() == DType::F16 {
                candle_core::quantized::GgmlDType::F16
            } else {
                candle_core::quantized::GgmlDType::F32
            };
            
            let q_tensor = QTensor::quantize(&tensor, target_type)?;
            out_tensors.push((name, q_tensor));
            num_skipped += 1;
        }
    }
    
    info!("Quantized {} tensors, kept {} tensors", num_quantized, num_skipped);

    // Save GGUF
    let mut output_file = std::fs::File::create(&options.output)?;
    
    // Prepare for GGUF write: need (&str, &QTensor)
    let out_tensors_refs: Vec<(&str, &QTensor)> = out_tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t))
        .collect();

    // ugg file writer
    gguf_file::write(&mut output_file, &[], &out_tensors_refs)?;

    info!(
        "Quantization complete. Saved to {:?} in {:.2?}",
        options.output,
        start.elapsed()
    );

    Ok(())
}
