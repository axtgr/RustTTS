# Qwen3-TTS Rust Engine

**Qwen3-TTS Rust Engine** — высокопроизводительный движок синтеза речи (TTS) на чистом Rust, полностью совместимый с моделью [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Цель проекта

Создать production-ready решение без зависимостей от Python и Torch Runtime, ориентированное на низкую задержку (low-latency) и эффективный стриминг (streaming).

## Ключевые особенности

- **Pure Rust:** Никакого Python в runtime
- **Производительность:** оптимизация под CPU/Metal/CUDA
- **Квантизация:** поддержка GGUF Q8/Q4 и safetensors
- **Streaming:** генерация и отдача аудио чанками по 20-100 мс
- **gRPC Server:** production-ready сервер с health checks и метриками
- **Modular Architecture:** 8 независимых crates

## Модель и веса

- Поддерживаются `model.safetensors`, `model.gguf`, `model-q8_0.gguf`, `model-q4_0.gguf`
- Приоритет выбора: `model.gguf` → `model-q8_0.gguf` → `model-q4_0.gguf` → `model.safetensors`
- Для CustomVoice используйте директорию `models/qwen3-tts-0.6b-customvoice`

## Быстрый старт

Проект использует protobuf, поэтому для сборки должен быть установлен [protoc](https://github.com/protocolbuffers/protobuf/releases).

### Apple Silicon (Metal) ускорение

Для владельцев Apple Silicon (включая M4) можно использовать feature-флаг `metal` для инференса на GPU (MPS).
См. `BENCHMARK.md`: на текущих реализациях Metal в бенчмарках уступает CPU, поэтому по умолчанию рекомендуется CPU.

```bash
# Сборка с поддержкой Metal
cargo build --release --features metal

# Синтез на GPU
cargo run -p tts-cli --release --features metal -- synth \
  --input "Тест GPU ускорения" \
  --output test_metal.wav \
  --model-dir models/qwen3-tts-0.6b-customvoice \
  --device metal
```

### CLI синтез

```bash
# Сборка (CPU)
cargo build --release

# Синтез текста в WAV (автоматический выбор устройства)
cargo run -p tts-cli --release -- synth \
  --input "Привет, мир!" \
  --model-dir models/qwen3-tts-0.6b-customvoice \
  -o output.wav

# Явное указание устройства
cargo run -p tts-cli --release --features metal -- synth \
  --input "Текст" \
  --model-dir models/qwen3-tts-0.6b-customvoice \
  --output out.wav \
  --device metal  # варианты: auto, cpu, metal, cuda

# Streaming режим
cargo run -p tts-cli --release -- synth \
  --input "Длинный текст..." \
  --model-dir models/qwen3-tts-0.6b-customvoice \
  -o output.wav \
  --streaming

# Нормализация текста (dry run)
cargo run -p tts-cli --release -- normalize --input "100 рублей" --lang ru
```

### Desktop App (GUI)

Приложение на базе Tauri v2.

```bash
# Установка Tauri CLI (если еще нет)
cargo install tauri-cli --version "^2.0.0"

# Запуск в режиме разработки (из папки crates/tts-app)
cd crates/tts-app
cargo tauri dev

# Запуск с поддержкой Metal
cargo tauri dev --features metal
```

### gRPC сервер

```bash
# Запуск сервера
cargo run -p tts-server --release

# Health check
curl http://localhost:8080/health

# gRPC синтез (требует grpcurl)
grpcurl -plaintext -d '{"text": "Привет мир", "language": 1}' \
  localhost:50051 tts.v1.TtsService/Synthesize
```

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        tts-cli / tts-server                      │
├─────────────────────────────────────────────────────────────────┤
│                            runtime                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ TtsPipeline │  │ StreamingSession │  │ BatchScheduler    │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  text-normalizer  │  text-tokenizer  │  acoustic-model  │ codec │
│  ┌─────────────┐  │  ┌────────────┐  │  ┌─────────────┐ │       │
│  │ Normalizer  │  │  │ Tokenizer  │  │  │ Transformer │ │       │
│  │ (RU/EN)     │  │  │ (HF/Mock)  │  │  │ + KV Cache  │ │       │
│  └─────────────┘  │  └────────────┘  │  └─────────────┘ │       │
├─────────────────────────────────────────────────────────────────┤
│                          tts-core                                │
│         Types, Traits, Errors, Config                            │
└─────────────────────────────────────────────────────────────────┘
```

## Структура проекта

| Crate | Описание |
|-------|----------|
| `tts-core` | Базовые типы, трейты, ошибки |
| `text-normalizer` | Нормализация текста (числа, даты, валюты) |
| `text-tokenizer` | BPE/Unigram токенизация |
| `acoustic-model` | Transformer модель с KV cache |
| `audio-codec-12hz` | Нейронный декодер аудио |
| `runtime` | Pipeline, streaming, batching |
| `tts-cli` | Командная строка |
| `tts-server` | gRPC + HTTP сервер |
| `tts-app` | Desktop приложение (Tauri) |

## Производительность

Сравнение с Python SDK (Apple Silicon, MPS vs Rust CPU/Metal):

| Метрика | Python SDK (MPS) | RustTTS (CPU, GGUF Q8) | RustTTS (Metal, GGUF Q8) |
|---------|------------------|------------------------|--------------------------|
| Загрузка модели | 7.7s | 1.92s | 2.20s |
| RTF (short) | 2.59x | 3.24x | 5.30x |
| RTF (medium) | 2.29x | 1.43x | 3.48x |
| RTF (long) | 1.95x | 1.37x | 3.29x |
| Размер | ~2GB (venv) | ~12MB | ~12MB |
| Cold start | ~7-10s | ~1.92s | ~2.20s |

> RTF (Real-Time Factor) — отношение времени синтеза к длительности аудио. Меньше = лучше.

Python быстрее на GPU для коротких запросов, Rust выигрывает на CPU на medium/long.

Подробности: **[Benchmark](BENCHMARK.md)**

## Тестирование

```bash
# Все тесты
cargo test --workspace

# Тесты конкретного crate
cargo test -p runtime

# С выводом
cargo test -- --nocapture

# Clippy
cargo clippy --workspace -- -D warnings
```

## Документация

- **[PRD (Требования)](docs/PRD.md)** — описание продукта, KPI
- **[Архитектура](docs/architecture.md)** — модули, потоки данных
- **[Roadmap](docs/roadmap.md)** — план развития
- **[План фаз](docs/phases/)** — детальные задачи

## Ссылки

- [Qwen3-TTS (оригинал)](https://github.com/QwenLM/Qwen3-TTS) — официальная модель от Alibaba
- [Candle](https://github.com/huggingface/candle) — ML framework на Rust
- [Tonic](https://github.com/hyperium/tonic) — gRPC для Rust

## Лицензия

MIT OR Apache-2.0
