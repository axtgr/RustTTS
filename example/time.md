# Тест 1 - квантированная модель 0.6b_Q8

## v1

Input:     34 chars
Language:  ru
Output:    output_rust_0.6b-customvoice-gguf.wav

Audio:
  Duration:    2.86 sec
  Samples:     68565
  Sample rate: 24000 Hz

Performance:
  Synthesis:   3395 ms
  Total:       6700 ms
  RTF:         1.189x
  Status:      Slower than real-time

## v2

Input:     34 chars
Language:  ru
Output:    output_rust_0.6b-customvoice-gguf_v2.wav

Audio:
  Duration:    2.38 sec
  Samples:     57045
  Sample rate: 24000 Hz

Performance:
  Synthesis:   2955 ms
  Total:       5289 ms
  RTF:         1.243x
  Status:      Slower than real-time

# Тест 3 - квантированная модель 1.7b_Q4

cargo run -p tts-cli --release -- synth \
  --input "Привет, Хабровчане" \
  --model-dir models/qwen3-tts-1.7b-customvoice-gguf \
  --codec-dir models/qwen3-tts-tokenizer \
  -o output_rust_1.7b-customvoice-gguf.wav

## v1
Input:     34 chars
Language:  ru
Output:    output_rust_1.7b-customvoice-gguf.wav

Audio:
  Duration:    1.18 sec
  Samples:     28245
  Sample rate: 24000 Hz

Performance:
  Synthesis:   1877 ms
  Total:       2687 ms
  RTF:         1.595x
  Status:      Slower than real-time

## v2

Input:     34 chars
Language:  ru
Output:    output_rust_1.7b-customvoice-gguf_v2.wav

Audio:
  Duration:    1.34 sec
  Samples:     32085
  Sample rate: 24000 Hz

Performance:
  Synthesis:   2083 ms
  Total:       4892 ms
  RTF:         1.558x
  Status:      Slower than real-time  

# Тест 4 - не квантированная модель 1.7b

cargo run -p tts-cli --release -- synth \
  --input "Привет, Хабровчане" \
  --model-dir models/qwen3-tts-1.7b-customvoice \
  --codec-dir models/qwen3-tts-tokenizer \
  -o output_rust_1.7b-customvoice.wav

Input:     34 chars
Language:  ru
Output:    o    utput_rust_1.7b-customvoice.wav

Audio:
  Duration:    1.58 sec
  Samples:     37845
  Sample rate: 24000 Hz

Performance:
  Synthesis:   10941 ms
  Total:       25138 ms
  RTF:         6.939x
  Status:      Slower than real-time