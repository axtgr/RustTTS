# Plan: Add Metal GPU Support (MPS) to Candle Backend

## Objective
Enable GPU acceleration on Apple Silicon devices (M1/M2/M3) using the Metal Performance Shaders (MPS) backend provided by the `candle` framework.

## Current State Analysis
1.  **Infrastructure Exists**: The `runtime` crate already contains device selection logic (`select_device`, `DevicePreference`) handling Metal, CUDA, and CPU.
2.  **Binary Configuration Missing**: The binary crates `tts-cli` and `tts-server` do not have the `metal` feature flag defined in their `Cargo.toml`. This prevents the necessary dependencies (like `candle-core/metal`) from being compiled with Metal support.
3.  **Hardcoded CPU**: The CLI tool (`tts-cli`) explicitly hardcodes `Device::Cpu` in its synthesis command, ignoring any potential GPU availability.

## Implementation Steps

### 1. Build Configuration (Cargo.toml)
Update binary crates to propagate feature flags correctly to dependencies.

*   **`crates/tts-cli/Cargo.toml`**:
    *   Add `metal` feature: `["runtime/metal", "acoustic-model/metal", "audio-codec-12hz/metal", "candle-core/metal"]`
    *   Update `cuda` feature to include `runtime/cuda` and `candle-core/cuda`.
*   **`crates/tts-server/Cargo.toml`**:
    *   Add `metal` feature similar to above.
    *   Update `cuda` feature.

### 2. CLI Refactoring (crates/tts-cli)
Enable user control over device selection.

*   **`src/main.rs`**:
    *   Add `--device` argument to the `Synth` command.
    *   Options: `auto` (default), `cpu`, `metal`, `cuda`.
*   **`src/commands/synth.rs`**:
    *   Update `SynthOptions` struct to include `device_preference`.
    *   Replace `let device = Device::Cpu;` with `runtime::device::select_device(options.device_preference)`.

## Verification
To verify the implementation, the project should be built with the `metal` feature and run with the device flag:

```bash
# Build
cargo build -p tts-cli --features metal

# Run
./target/debug/tts synth --input "Hello" --output test.wav --device metal
```

Expected behavior:
*   Logs should indicate "Using Metal GPU (Apple Silicon)".
*   Inference should run on the GPU.
