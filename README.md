# Qwen3-TTS Rust Engine

**Qwen3-TTS Rust Engine** — высокопроизводительный движок синтеза речи (TTS) на чистом Rust, полностью совместимый с моделью [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Цель проекта

Создать production-ready решение без зависимостей от Python и Torch Runtime, ориентированное на низкую задержку (low-latency) и эффективный стриминг (streaming).

## Ключевые особенности

- **Pure Rust:** Никакого Python в runtime
- **High Performance:** P95 latency < 400ms (GPU), поддержка CUDA
- **Streaming:** Генерация и отдача аудио чанками по 20-100 мс
- **gRPC Server:** Production-ready сервер с health checks и метриками
- **Modular Architecture:** 8 независимых crates

## Быстрый старт

### CLI синтез

```bash
# Сборка
cargo build --release

# Синтез текста в WAV
cargo run -p tts-cli --release -- synth "Привет, мир!" -o output.wav

# Streaming режим
cargo run -p tts-cli --release -- synth "Длинный текст для синтеза..." -o output.wav --streaming

# Нормализация текста (dry run)
cargo run -p tts-cli --release -- normalize "100 рублей" --lang ru
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

## Статус разработки

| Фаза | Статус | Описание |
|------|--------|----------|
| Phase 0 | ✅ | Инфраструктура workspace |
| Phase 1 | ✅ | Text Pipeline (normalizer, tokenizer) |
| Phase 2 | ✅ | Acoustic Model (transformer, KV cache) |
| Phase 3 | ✅ | Audio Codec (neural decoder, streaming) |
| Phase 4 | ✅ | Runtime (pipeline, batching) |
| Phase 5 | ✅ | CLI (synth, normalize, tokenize) |
| Phase 6 | ✅ | gRPC Server (streaming, health, metrics) |

**Текущий статус:** Все основные фазы завершены. Полнофункциональный TTS pipeline работает.

## Производительность

Сравнение с Python SDK (Apple Silicon, MPS vs CPU):

| Метрика | Python SDK (MPS) | RustTTS (CPU) |
|---------|------------------|---------------|
| Загрузка модели | 2.4s | 2.7s |
| RTF (short text) | 4.3x | 5.4x |
| RTF (long text) | 2.5x | 3.6x |
| Размер | ~2GB (venv) | ~12MB |
| Cold start | ~5-10s | ~3s |

> RTF (Real-Time Factor) — отношение времени синтеза к длительности аудио. Меньше = лучше.

Python быстрее за счет GPU ускорения (MPS). Rust выигрывает в размере и простоте развертывания.

Подробности: **[Benchmark](docs/benchmark.md)**

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
