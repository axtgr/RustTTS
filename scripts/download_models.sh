#!/usr/bin/env bash
# Download Qwen3-TTS models from HuggingFace
#
# Usage:
#   ./scripts/download_models.sh [--tiny|--base|--all]
#
# Requirements:
#   - huggingface-cli installed: pip install huggingface_hub
#   - Or: brew install huggingface-cli

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
    if ! command -v huggingface-cli &> /dev/null; then
        log_error "huggingface-cli not found!"
        echo "Install with: pip install huggingface_hub"
        echo "Or: brew install huggingface-cli"
        exit 1
    fi
    log_info "huggingface-cli found"
}

download_model() {
    local model_name="$1"
    local local_dir="$2"
    
    log_info "Downloading $model_name to $local_dir..."
    
    mkdir -p "$local_dir"
    
    local args=(
        download
        "$model_name"
        --local-dir "$local_dir"
        --local-dir-use-symlinks False
    )
    
    if [[ -n "$HF_TOKEN" ]]; then
        args+=(--token "$HF_TOKEN")
    fi
    
    if huggingface-cli "${args[@]}"; then
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
    echo "  --talker     Download only the Talker model (1.8GB)"
    echo "  --tokenizer  Download only the Audio Tokenizer model (682MB)"
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
    
    if $download_talker_flag; then
        download_talker
    fi
    
    if $download_tokenizer_flag; then
        download_tokenizer
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
}

main "$@"
