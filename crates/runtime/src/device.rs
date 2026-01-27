//! Device selection for GPU acceleration.
//!
//! This module provides automatic device selection with fallback:
//! Metal (Apple Silicon) → CUDA (NVIDIA) → CPU

use candle_core::Device;
use tracing::{info, warn};

use tts_core::{TtsError, TtsResult};

/// Device preference for model loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Automatically select the best available device.
    #[default]
    Auto,
    /// Force CPU usage.
    Cpu,
    /// Force Metal GPU (Apple Silicon).
    Metal,
    /// Force CUDA GPU (NVIDIA).
    Cuda,
}

impl DevicePreference {
    /// Parse from string (for CLI/config).
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "metal" | "mps" | "apple" => Self::Metal,
            "cuda" | "gpu" | "nvidia" => Self::Cuda,
            _ => Self::Auto,
        }
    }
}

/// Select the best available device based on preference and compiled features.
///
/// # Arguments
/// * `preference` - Device preference (Auto, Cpu, Metal, Cuda)
///
/// # Returns
/// * `Ok(Device)` - Selected device
/// * `Err` - If preferred device is not available
pub fn select_device(preference: DevicePreference) -> TtsResult<Device> {
    match preference {
        DevicePreference::Cpu => {
            info!("Using CPU device (forced)");
            Ok(Device::Cpu)
        }
        DevicePreference::Metal => select_metal(),
        DevicePreference::Cuda => select_cuda(),
        DevicePreference::Auto => select_auto(),
    }
}

/// Automatically select the best available device.
fn select_auto() -> TtsResult<Device> {
    // Try Metal first (Apple Silicon)
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Auto-selected Metal GPU (Apple Silicon)");
                return Ok(device);
            }
            Err(e) => {
                warn!("Metal GPU not available: {}", e);
            }
        }
    }

    // Try CUDA (NVIDIA)
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Auto-selected CUDA GPU (NVIDIA)");
                return Ok(device);
            }
            Err(e) => {
                warn!("CUDA GPU not available: {}", e);
            }
        }
    }

    // Fallback to CPU
    info!("Using CPU device (no GPU available)");
    Ok(Device::Cpu)
}

/// Try to select Metal device.
#[allow(unused_variables)]
fn select_metal() -> TtsResult<Device> {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Using Metal GPU (Apple Silicon)");
                Ok(device)
            }
            Err(e) => Err(TtsError::config(format!(
                "Metal GPU requested but not available: {}",
                e
            ))),
        }
    }

    #[cfg(not(feature = "metal"))]
    {
        Err(TtsError::config(
            "Metal GPU requested but 'metal' feature not enabled. \
             Rebuild with: cargo build --features metal"
                .to_string(),
        ))
    }
}

/// Try to select CUDA device.
#[allow(unused_variables)]
fn select_cuda() -> TtsResult<Device> {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using CUDA GPU (NVIDIA)");
                Ok(device)
            }
            Err(e) => Err(TtsError::config(format!(
                "CUDA GPU requested but not available: {}",
                e
            ))),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(TtsError::config(
            "CUDA GPU requested but 'cuda' feature not enabled. \
             Rebuild with: cargo build --features cuda"
                .to_string(),
        ))
    }
}

/// Get device name for logging/display.
pub fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA GPU",
        Device::Metal(_) => "Metal GPU",
    }
}

/// Check if Metal feature is enabled.
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).is_ok()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Check if CUDA feature is enabled.
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_preference_from_str() {
        assert_eq!(DevicePreference::from_str("cpu"), DevicePreference::Cpu);
        assert_eq!(DevicePreference::from_str("CPU"), DevicePreference::Cpu);
        assert_eq!(DevicePreference::from_str("metal"), DevicePreference::Metal);
        assert_eq!(DevicePreference::from_str("Metal"), DevicePreference::Metal);
        assert_eq!(DevicePreference::from_str("mps"), DevicePreference::Metal);
        assert_eq!(DevicePreference::from_str("cuda"), DevicePreference::Cuda);
        assert_eq!(DevicePreference::from_str("gpu"), DevicePreference::Cuda);
        assert_eq!(DevicePreference::from_str("auto"), DevicePreference::Auto);
        assert_eq!(
            DevicePreference::from_str("unknown"),
            DevicePreference::Auto
        );
    }

    #[test]
    fn test_select_cpu() {
        let device = select_device(DevicePreference::Cpu).unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_select_auto() {
        // Should always succeed (falls back to CPU)
        let device = select_device(DevicePreference::Auto).unwrap();
        // Result depends on available hardware
        assert!(matches!(
            device,
            Device::Cpu | Device::Metal(_) | Device::Cuda(_)
        ));
    }

    #[test]
    fn test_device_name() {
        assert_eq!(device_name(&Device::Cpu), "CPU");
    }
}
