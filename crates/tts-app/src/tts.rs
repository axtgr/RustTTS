//! TTS commands for Tauri application.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use rodio::{Decoder, OutputStream, Sink};
use serde::{Deserialize, Serialize};
use tauri::State;
use tokio::sync::Mutex;
use tracing::{error, info};

use audio_codec_12hz::apply_fade_in;
use runtime::TtsPipeline;
use tts_core::Lang;

/// Application state managed by Tauri.
pub struct AppState {
    /// TTS pipeline (lazy initialized).
    pipeline: Mutex<Option<Arc<TtsPipeline>>>,
    /// Current model directory.
    model_dir: Mutex<Option<PathBuf>>,
    /// Flag to stop playback.
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            pipeline: Mutex::new(None),
            model_dir: Mutex::new(None),
            stop_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
}

/// TTS initialization parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitParams {
    /// Path to model directory.
    pub model_dir: String,
}

/// TTS synthesis parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakParams {
    /// Text to synthesize.
    pub text: String,
    /// Speaker name (e.g., "ryan", "vivian").
    pub speaker: Option<String>,
    /// Language code (e.g., "ru", "en").
    pub language: Option<String>,
    /// Temperature for sampling (0.0-2.0).
    pub temperature: Option<f64>,
    /// Top-p for nucleus sampling (0.0-1.0).
    pub top_p: Option<f64>,
}

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub has_speakers: bool,
}

/// Language information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
}

/// Synthesis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub success: bool,
    pub duration_ms: u64,
    pub sample_rate: u32,
    pub num_samples: usize,
    pub error: Option<String>,
}

/// Initialize TTS pipeline with specified model.
#[tauri::command]
pub async fn init_tts(params: InitParams, state: State<'_, AppState>) -> Result<String, String> {
    info!("Initializing TTS with model: {}", params.model_dir);

    let model_path = PathBuf::from(&params.model_dir);
    if !model_path.exists() {
        return Err(format!("Model directory not found: {}", params.model_dir));
    }

    // Text tokenizer is inside the model directory (vocab.json, merges.txt)
    let tokenizer_dir = model_path.clone();

    // Audio codec (Qwen3-TTS-Tokenizer-12Hz) - check multiple locations
    let codec_dir = {
        // First try speech_tokenizer inside model
        let speech_tok = model_path.join("speech_tokenizer");
        if speech_tok.exists() {
            speech_tok
        } else {
            // Then try separate qwen3-tts-tokenizer directory
            let parent_tok = model_path.parent().unwrap().join("qwen3-tts-tokenizer");
            if parent_tok.exists() {
                parent_tok
            } else {
                return Err(format!(
                    "Audio codec not found. Checked: {} and {}",
                    speech_tok.display(),
                    parent_tok.display()
                ));
            }
        }
    };

    info!("Model: {}", model_path.display());
    info!("Tokenizer: {}", tokenizer_dir.display());
    info!("Codec: {}", codec_dir.display());

    // Use CPU device (Metal/CUDA support can be added later)
    let device = Device::Cpu;

    // Load pipeline
    let pipeline =
        match TtsPipeline::from_pretrained(&model_path, &tokenizer_dir, &codec_dir, &device) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to load TTS pipeline: {}", e);
                return Err(format!("Failed to load model: {}", e));
            }
        };

    // Store pipeline in state
    {
        let mut pipeline_lock = state.pipeline.lock().await;
        *pipeline_lock = Some(Arc::new(pipeline));
    }
    {
        let mut model_dir_lock = state.model_dir.lock().await;
        *model_dir_lock = Some(model_path);
    }

    info!("TTS initialized successfully");
    Ok("TTS initialized successfully".to_string())
}

/// Synthesize and play text.
#[tauri::command]
pub async fn speak(
    params: SpeakParams,
    state: State<'_, AppState>,
) -> Result<SynthesisResult, String> {
    let start = std::time::Instant::now();
    info!("Synthesizing: {} chars", params.text.len());

    // Reset stop flag
    state
        .stop_flag
        .store(false, std::sync::atomic::Ordering::SeqCst);

    // Get pipeline
    let pipeline = {
        let lock = state.pipeline.lock().await;
        match lock.as_ref() {
            Some(p) => Arc::clone(p),
            None => return Err("TTS not initialized. Call init_tts first.".to_string()),
        }
    };

    // Parse language (currently only Ru, En, Mixed supported)
    let lang = params
        .language
        .as_ref()
        .map(|l| match l.to_lowercase().as_str() {
            "ru" | "russian" => Lang::Ru,
            "en" | "english" => Lang::En,
            "mixed" | "auto" => Lang::Mixed,
            _ => Lang::En,
        });

    // Synthesize using the pipeline's speaker-aware method
    let audio =
        match pipeline.synthesize_with_speaker(&params.text, lang, params.speaker.as_deref()) {
            Ok(a) => a,
            Err(e) => {
                error!("Synthesis failed: {}", e);
                return Ok(SynthesisResult {
                    success: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    sample_rate: 0,
                    num_samples: 0,
                    error: Some(e.to_string()),
                });
            }
        };

    let sample_rate = audio.sample_rate;
    let num_samples = audio.num_samples();

    // Apply fade-in to remove artifacts
    // audio.pcm is Arc<[f32]>, need to clone to mutable Vec
    let mut samples: Vec<f32> = audio.pcm.to_vec();
    apply_fade_in(&mut samples, 50.0, sample_rate);

    // Create WAV in memory
    let wav_data = create_wav_buffer(&samples, sample_rate);

    let duration_ms = start.elapsed().as_millis() as u64;
    info!(
        "Synthesis complete: {} samples, {}ms",
        num_samples, duration_ms
    );

    // Play audio in a separate thread (rodio is not Send)
    let stop_flag = Arc::clone(&state.stop_flag);
    std::thread::spawn(move || {
        if let Err(e) = play_audio_blocking(wav_data, stop_flag) {
            error!("Audio playback error: {}", e);
        }
    });

    Ok(SynthesisResult {
        success: true,
        duration_ms,
        sample_rate,
        num_samples,
        error: None,
    })
}

/// Play audio in blocking mode.
fn play_audio_blocking(
    wav_data: Vec<u8>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    let (_stream, handle) =
        OutputStream::try_default().map_err(|e| format!("Failed to get audio output: {}", e))?;

    let sink = Sink::try_new(&handle).map_err(|e| format!("Failed to create audio sink: {}", e))?;

    let cursor = Cursor::new(wav_data);
    let source = Decoder::new(cursor).map_err(|e| format!("Failed to decode audio: {}", e))?;

    sink.append(source);
    sink.play();

    // Wait for playback to complete or stop signal
    while !sink.empty() {
        if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            sink.stop();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    Ok(())
}

/// Stop current audio playback.
#[tauri::command]
pub async fn stop_audio(state: State<'_, AppState>) -> Result<(), String> {
    state
        .stop_flag
        .store(true, std::sync::atomic::Ordering::SeqCst);
    info!("Audio playback stop requested");
    Ok(())
}

/// Get available models.
#[tauri::command]
pub async fn get_available_models() -> Result<Vec<ModelInfo>, String> {
    // Look for models in common locations
    let mut models = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();

    // Check relative to executable
    let exe_path = std::env::current_exe().ok();
    let exe_dir = exe_path
        .as_ref()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    // Get current working directory
    let cwd = std::env::current_dir().ok();

    info!("Searching for models...");
    info!("  Executable: {:?}", exe_path);
    info!("  Exe dir: {:?}", exe_dir);
    info!("  CWD: {:?}", cwd);

    let mut search_paths = vec![
        PathBuf::from("models"),
        PathBuf::from("../models"),
        PathBuf::from("../../models"),
    ];

    if let Some(ref exe) = exe_dir {
        search_paths.push(exe.join("models"));
        search_paths.push(exe.join("../Resources/models"));
        // For development: go up from target/release or target/debug
        search_paths.push(exe.join("../../models"));
        search_paths.push(exe.join("../../../models"));
        search_paths.push(exe.join("../../../../models")); // deeper for nested builds
    }

    if let Some(ref cwd) = cwd {
        search_paths.push(cwd.join("models"));
    }

    // Add home directory common locations
    if let Some(home) = dirs::home_dir() {
        search_paths.push(home.join("Projects/RustTTS/models"));
        search_paths.push(home.join("models"));
    }

    for base_path in &search_paths {
        info!(
            "  Checking: {:?} (exists: {})",
            base_path,
            base_path.exists()
        );

        if !base_path.exists() {
            continue;
        }

        // Canonicalize to avoid duplicates
        let canonical = match base_path.canonicalize() {
            Ok(c) => c,
            Err(_) => continue,
        };

        if seen_paths.contains(&canonical) {
            continue;
        }
        seen_paths.insert(canonical);

        if let Ok(entries) = std::fs::read_dir(base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let config_path = path.join("config.json");
                    if config_path.exists() {
                        let name = path.file_name().unwrap().to_string_lossy().to_string();
                        // Skip tokenizer directory
                        if name.contains("tokenizer") {
                            continue;
                        }
                        let has_speakers = name.to_lowercase().contains("customvoice");
                        let full_path = path
                            .canonicalize()
                            .unwrap_or(path.clone())
                            .to_string_lossy()
                            .to_string();
                        info!("  Found model: {} at {}", name, full_path);
                        models.push(ModelInfo {
                            name: name.clone(),
                            path: full_path,
                            has_speakers,
                        });
                    }
                }
            }
        }
    }

    info!("Found {} models total", models.len());
    Ok(models)
}

/// Get available speakers for current model.
#[tauri::command]
pub async fn get_available_speakers(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let model_dir = state.model_dir.lock().await;

    match model_dir.as_ref() {
        Some(path) => {
            // Read config.json to get speakers
            let config_path = path.join("config.json");
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(spk_id) = config
                        .get("talker_config")
                        .and_then(|t| t.get("spk_id"))
                        .and_then(|s| s.as_object())
                    {
                        let speakers: Vec<String> = spk_id.keys().cloned().collect();
                        return Ok(speakers);
                    }
                }
            }
            Ok(vec![])
        }
        None => Ok(vec![]),
    }
}

/// Get available languages.
#[tauri::command]
pub async fn get_available_languages() -> Result<Vec<LanguageInfo>, String> {
    // Currently only Ru, En, Mixed are supported in Lang enum
    Ok(vec![
        LanguageInfo {
            code: "ru".to_string(),
            name: "Russian".to_string(),
        },
        LanguageInfo {
            code: "en".to_string(),
            name: "English".to_string(),
        },
        LanguageInfo {
            code: "mixed".to_string(),
            name: "Auto-detect".to_string(),
        },
    ])
}

/// Create WAV buffer from samples.
fn create_wav_buffer(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut buffer = Vec::new();

    // WAV header
    let data_size = (samples.len() * 2) as u32; // 16-bit samples
    let file_size = 36 + data_size;

    // RIFF header
    buffer.extend_from_slice(b"RIFF");
    buffer.extend_from_slice(&file_size.to_le_bytes());
    buffer.extend_from_slice(b"WAVE");

    // fmt chunk
    buffer.extend_from_slice(b"fmt ");
    buffer.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buffer.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buffer.extend_from_slice(&1u16.to_le_bytes()); // mono
    buffer.extend_from_slice(&sample_rate.to_le_bytes());
    buffer.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buffer.extend_from_slice(&2u16.to_le_bytes()); // block align
    buffer.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buffer.extend_from_slice(b"data");
    buffer.extend_from_slice(&data_size.to_le_bytes());

    // Convert f32 samples to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        buffer.extend_from_slice(&i16_sample.to_le_bytes());
    }

    buffer
}
