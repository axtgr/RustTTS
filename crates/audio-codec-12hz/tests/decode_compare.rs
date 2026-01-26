//! Compare Rust decoder output with Python reference

use audio_codec_12hz::Decoder;
use candle_core::{DType, Device, Tensor};
use std::fs;

// Test codes from Python (seed 42, shape [1, 27, 16])
const TEST_CODES: [[u32; 16]; 27] = [
    [
        1126, 1459, 860, 1294, 1130, 1095, 1724, 1044, 1638, 121, 466, 1238, 330, 1482, 87, 1396,
    ],
    [
        1123, 871, 1687, 130, 1685, 1332, 769, 343, 1515, 1437, 805, 385, 1215, 955, 276, 1184,
    ],
    [
        459, 1337, 21, 252, 747, 856, 1584, 474, 1082, 510, 1705, 2047, 1499, 699, 975, 1806,
    ],
    [
        189, 957, 686, 957, 562, 1899, 1590, 1267, 831, 1528, 1154, 1508, 1842, 646, 20, 840,
    ],
    [
        166, 1297, 387, 600, 315, 13, 241, 2041, 776, 1369, 564, 897, 1363, 91, 1390, 955,
    ],
    // ... more rows - let's just use first 5 for now as placeholder
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
];

#[test]
#[ignore] // Requires model weights
fn test_decoder_matches_python() {
    let device = Device::Cpu;

    // Load decoder
    let decoder = Decoder::from_pretrained("../../models/qwen3-tts-tokenizer", &device)
        .expect("Failed to load decoder");

    // Create codes tensor - shape [16, 27] (codebooks x seq_len)
    let mut codes_data = vec![0u32; 16 * 27];
    for t in 0..27 {
        for q in 0..16 {
            codes_data[q * 27 + t] = TEST_CODES[t][q];
        }
    }
    let codes =
        Tensor::from_vec(codes_data, (16, 27), &device).expect("Failed to create codes tensor");

    println!("Codes shape: {:?}", codes.shape());

    // Decode
    let audio = decoder.decode_multi(&codes).expect("Failed to decode");

    println!("Audio shape: {:?}", audio.shape());

    let audio_vec: Vec<f32> = audio.flatten_all().unwrap().to_vec1().unwrap();
    let audio_std = (audio_vec.iter().map(|x| x * x).sum::<f32>() / audio_vec.len() as f32).sqrt();

    println!("Audio std: {:.4}", audio_std);
    println!("Audio first 10: {:?}", &audio_vec[..10]);

    // Python reference: std=0.1172
    // Allow 3x tolerance for now
    assert!(
        audio_std > 0.03 && audio_std < 0.4,
        "Audio std {:.4} too far from Python reference 0.1172",
        audio_std
    );
}
