//! TTS Desktop Application Library
//!
//! Provides Tauri commands for text-to-speech synthesis.

mod tts;

pub use tts::{
    get_available_languages, get_available_models, get_available_speakers, init_tts, speak,
    stop_audio, AppState,
};

/// Run the Tauri application.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tts_app=info".parse().unwrap())
                .add_directive("runtime=info".parse().unwrap()),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            init_tts,
            speak,
            stop_audio,
            get_available_models,
            get_available_speakers,
            get_available_languages,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
