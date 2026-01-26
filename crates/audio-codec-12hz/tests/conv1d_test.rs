use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use safetensors::SafeTensors;
use std::fs;
use std::path::PathBuf;

fn find_model_file() -> PathBuf {
    let candidates = [
        "models/qwen3-tts-tokenizer/model.safetensors",
        "../../../models/qwen3-tts-tokenizer/model.safetensors",
        "../../models/qwen3-tts-tokenizer/model.safetensors",
    ];
    for candidate in &candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let path =
            PathBuf::from(manifest_dir).join("../../models/qwen3-tts-tokenizer/model.safetensors");
        if path.exists() {
            return path;
        }
    }
    panic!("Could not find model file");
}

#[test]
fn test_conv1d_with_randn_input() -> Result<()> {
    let device = Device::Cpu;

    let model_path = find_model_file();
    println!("Using model: {:?}", model_path);

    let bytes = fs::read(&model_path).expect("read model");
    let safetensors = SafeTensors::deserialize(&bytes).expect("parse");

    let w_view = safetensors
        .tensor("decoder.decoder.6.conv.weight")
        .expect("weight");
    let b_view = safetensors
        .tensor("decoder.decoder.6.conv.bias")
        .expect("bias");

    let weight = Tensor::from_raw_buffer(w_view.data(), DType::F32, &[1, 96, 7], &device)?;
    let bias = Tensor::from_raw_buffer(b_view.data(), DType::F32, &[1], &device)?;

    // Create randn-like input with std=25
    // Using Box-Muller transform for reproducibility
    use std::f32::consts::PI;
    let n = 96 * 51853;
    let mut input_data = Vec::with_capacity(n);

    // Simple LCG random number generator with seed 42
    let mut rng_state: u64 = 42;
    let lcg_next = |state: &mut u64| -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (u32::MAX as f32)
    };

    for _ in 0..(n / 2) {
        let u1 = lcg_next(&mut rng_state).max(1e-10);
        let u2 = lcg_next(&mut rng_state);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
        input_data.push(z0 * 25.0);
        input_data.push(z1 * 25.0);
    }
    if input_data.len() < n {
        let u1 = lcg_next(&mut rng_state).max(1e-10);
        let u2 = lcg_next(&mut rng_state);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        input_data.push(z0 * 25.0);
    }
    input_data.truncate(n);

    let input = Tensor::from_vec(input_data, (1, 96, 51853), &device)?;

    let input_std = input.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!("Input RMS (std): {:.4}", input_std);

    // Print first values for channels 0, 1, 95
    let x_ch0: Vec<f32> = input.i((0, 0, 0..10))?.to_vec1()?;
    let x_ch1: Vec<f32> = input.i((0, 1, 0..10))?.to_vec1()?;
    let x_ch95: Vec<f32> = input.i((0, 95, 0..10))?.to_vec1()?;
    println!("x[0,0,:10]: {:?}", x_ch0);
    println!("x[0,1,:10]: {:?}", x_ch1);
    println!("x[0,95,:10]: {:?}", x_ch95);

    // Do convolution
    let conv = Conv1d::new(
        weight,
        Some(bias),
        Conv1dConfig {
            padding: 3,
            ..Default::default()
        },
    );
    let output = conv.forward(&input)?;

    println!("\nOutput shape: {:?}", output.shape());
    let output_std = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!("Output RMS (std): {:.4}", output_std);

    // Get min/max
    let out_flat: Vec<f32> = output.flatten_all()?.to_vec1()?;
    let out_min = out_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let out_max = out_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Output range: [{:.4}, {:.4}]", out_min, out_max);

    // Expected: ~0.21 based on Python with std=25 input
    println!("\nExpected output std: ~0.21");
    let ratio = output_std / 0.21;
    println!("Ratio (actual/expected): {:.2}", ratio);

    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Output std {:.4} too far from 0.21",
        output_std
    );

    Ok(())
}
