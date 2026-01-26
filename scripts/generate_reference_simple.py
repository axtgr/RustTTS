#!/usr/bin/env python3
"""
Generate reference audio using Qwen3-TTS model directly without full SDK.
Requires: torch, transformers, soundfile
"""

import json
import sys
from pathlib import Path

import torch
import soundfile as sf

# Add Qwen3-TTS to path
sys.path.insert(0, "/tmp/Qwen3-TTS")

# Config
MODEL_PATH = "models/qwen3-tts-0.6b-customvoice"
TEXT = "Hello world"
SPEAKER = "ryan"
LANG = "english"
OUTPUT_DIR = Path("tests/golden")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")

    # Import Qwen3-TTS components
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models import Qwen3TTSProcessor

    # Load config
    config = Qwen3TTSConfig.from_pretrained(MODEL_PATH)
    print(f"Config loaded: {config.tts_model_type}")

    # Load model
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Load processor/tokenizer
    processor = Qwen3TTSProcessor.from_pretrained(MODEL_PATH)

    print(f"Model loaded successfully")
    print(f"  tts_model_type: {model.tts_model_type}")
    print(f"  tokenizer_type: {model.tokenizer_type}")

    # Build input text in CustomVoice format
    assistant_text = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\nInput text: {assistant_text!r}")

    # Tokenize
    inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")

    # Get special tokens
    talker_config = config.talker_config
    print(f"\nSpecial tokens:")
    print(f"  tts_bos: {config.tts_bos_token_id}")
    print(f"  tts_eos: {config.tts_eos_token_id}")
    print(f"  tts_pad: {config.tts_pad_token_id}")
    print(f"  codec_bos: {talker_config.codec_bos_id}")
    print(f"  codec_eos: {talker_config.codec_eos_token_id}")
    print(f"  codec_pad: {talker_config.codec_pad_id}")
    print(f"  codec_think: {talker_config.codec_think_id}")
    print(f"  codec_think_bos: {talker_config.codec_think_bos_id}")
    print(f"  codec_think_eos: {talker_config.codec_think_eos_id}")

    # Language and speaker IDs
    lang_id = talker_config.codec_language_id.get(LANG.lower())
    spk_id = talker_config.spk_id.get(SPEAKER.lower())
    print(f"\n  Language '{LANG}' ID: {lang_id}")
    print(f"  Speaker '{SPEAKER}' ID: {spk_id}")

    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        talker_codes_list, hidden_states_list = model.generate(
            input_ids=[input_ids],
            instruct_ids=None,
            languages=[LANG],
            speakers=[SPEAKER],
            non_streaming_mode=True,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
        )

    print(f"Generated {len(talker_codes_list)} sequences")

    if talker_codes_list:
        codes = talker_codes_list[0]
        print(f"Codes shape: {codes.shape}")
        print(f"First 20 codes: {codes[:20].tolist()}")
        print(f"Last 10 codes: {codes[-10:].tolist()}")

        # Save codes
        codes_path = OUTPUT_DIR / "reference_codes.json"
        with open(codes_path, "w") as f:
            json.dump(
                {
                    "text": TEXT,
                    "speaker": SPEAKER,
                    "language": LANG,
                    "input_ids": input_ids.tolist(),
                    "codes": codes.tolist(),
                    "codes_shape": list(codes.shape),
                },
                f,
                indent=2,
            )
        print(f"\nSaved codes to {codes_path}")

        # Decode to audio using audio codec
        print("\nDecoding to audio...")
        # This part needs the audio tokenizer which may not be available
        # For now just save the codes


if __name__ == "__main__":
    main()
