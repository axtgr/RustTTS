#!/usr/bin/env bash
# Download Qwen3-TTS models from HuggingFace
#
# Usage:
#   ./scripts/download_models.sh [--talker|--tokenizer|--all]
#
# Requirements:
#   - Python 3.8+ with huggingface_hub: pip install huggingface_hub

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
HF_TOKEN="${HF_TOKEN:-}"

# Model names on HuggingFace
TALKER_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TOKENIZER_MODEL="Qwen/Qwen3-TTS-Tokenizer-12Hz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    if ! python3 -c "import huggingface_hub" &> /dev/null; then
        log_error "huggingface_hub not found!"
        echo "Install with: pip install huggingface_hub"
        echo "Or: pip3 install --user huggingface_hub"
        exit 1
    fi
    log_info "huggingface_hub found"
}

download_model() {
    local model_name="$1"
    local local_dir="$2"
    
    log_info "Downloading $model_name to $local_dir..."
    log_info "This may take a while for large models..."
    
    mkdir -p "$local_dir"
    
    python3 << EOF
from huggingface_hub import snapshot_download
import os

token = os.environ.get('HF_TOKEN', None) or None
snapshot_download(
    repo_id="$model_name",
    local_dir="$local_dir",
    token=token
)
print("Download complete!")
EOF
    
    if [[ $? -eq 0 ]]; then
        log_info "Successfully downloaded $model_name"
    else
        log_error "Failed to download $model_name"
        return 1
    fi
}

download_talker() {
    local dir="$MODELS_DIR/qwen3-tts-0.6b"
    
    if [[ -f "$dir/model.safetensors" ]]; then
        log_warn "Talker model already exists at $dir"
        read -p "Re-download? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    download_model "$TALKER_MODEL" "$dir"
}

download_tokenizer() {
    local dir="$MODELS_DIR/qwen3-tts-tokenizer"
    
    if [[ -f "$dir/model.safetensors" ]]; then
        log_warn "Tokenizer model already exists at $dir"
        read -p "Re-download? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    download_model "$TOKENIZER_MODEL" "$dir"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --talker     Download only the Talker model (~1.8GB)"
    echo "  --tokenizer  Download only the Audio Tokenizer model (~682MB)"
    echo "  --all        Download all models (default)"
    echo "  --help       Show this help message"
    echo
    echo "Environment variables:"
    echo "  MODELS_DIR   Directory to store models (default: ./models)"
    echo "  HF_TOKEN     HuggingFace token for gated models"
    echo
    echo "Models:"
    echo "  Talker:    $TALKER_MODEL (~1.8GB)"
    echo "  Tokenizer: $TOKENIZER_MODEL (~682MB)"
    echo
    echo "After download, run TTS with:"
    echo "  cargo run --bin tts -- synth \"Hello\" -o out.wav \\"
    echo "    --model-dir models/qwen3-tts-0.6b \\"
    echo "    --codec-dir models/qwen3-tts-tokenizer"
}

main() {
    local download_talker_flag=false
    local download_tokenizer_flag=false
    
    # Parse arguments
    if [[ $# -eq 0 ]]; then
        download_talker_flag=true
        download_tokenizer_flag=true
    else
        for arg in "$@"; do
            case $arg in
                --talker)
                    download_talker_flag=true
                    ;;
                --tokenizer)
                    download_tokenizer_flag=true
                    ;;
                --all)
                    download_talker_flag=true
                    download_tokenizer_flag=true
                    ;;
                --help|-h)
                    show_usage
                    exit 0
                    ;;
                *)
                    log_error "Unknown option: $arg"
                    show_usage
                    exit 1
                    ;;
            esac
        done
    fi
    
    check_dependencies
    
    log_info "Models will be downloaded to: $MODELS_DIR"
    mkdir -p "$MODELS_DIR"
    
    if $download_tokenizer_flag; then
        download_tokenizer
    fi
    
    if $download_talker_flag; then
        download_talker
    fi
    
    log_info "Done!"
    echo
    echo "Model locations:"
    if $download_talker_flag; then
        echo "  Talker:    $MODELS_DIR/qwen3-tts-0.6b/"
    fi
    if $download_tokenizer_flag; then
        echo "  Tokenizer: $MODELS_DIR/qwen3-tts-tokenizer/"
    fi
    echo
    echo "Run TTS with:"
    echo "  cargo run --bin tts -- synth \"Привет мир\" -o out.wav \\"
    echo "    --model-dir $MODELS_DIR/qwen3-tts-0.6b \\"
    echo "    --codec-dir $MODELS_DIR/qwen3-tts-tokenizer"
}

main "$@"
