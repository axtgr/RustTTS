//! TTS Desktop Application Entry Point

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

fn main() {
    tts_app_lib::run();
}
