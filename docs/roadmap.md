# Roadmap: Qwen3-TTS Rust Engine

## Цели

- Rust-only реализация Qwen3-TTS (без Python, без torch)
- Минимальная latency до first audio; RTF < 0.2 на GPU
- Streaming чанки 20–100 мс с overlap/crossfade
- Поддержка CPU/GPU, CUDA backend, батчинг и QoS
- Весы в safetensors, воспроизводимый build через `cargo`

## Основные KPI

- Time-to-first-audio (GPU): 150–400 мс; CPU: 600–1500 мс
- RTF: GPU < 0.2; CPU 0.8–2.0 (зависит от железа)
- Длительная стабильность: 10+ минут стрима без утечек
- Детерминизм: опция фиксированного seed

## Фазы

| Фаза | Название | Статус | Описание |
|------|----------|--------|----------|
| Ф0 | Инфраструктура | ✅ | Workspace, конфиги, CI, базовые типы |
| Ф1 | Текстовый пайплайн | ✅ | Нормализация RU/EN, токенизация |
| Ф2 | Акустическая модель | ✅ | Трансформер, KV cache, streaming токены |
| Ф3 | Аудио-декодер | ✅ | Qwen3-TTS-Tokenizer-12Hz → PCM, чанки, crossfade |
| Ф4 | Runtime | ✅ | Pipeline интеграция, батчинг, QoS, метрики |
| Ф5 | CLI | ✅ | Synth команда, WAV вывод, streaming режим |
| Ф6 | gRPC Server | ✅ | Streaming API, health checks, Prometheus |
| Ф7 | Оптимизация | ⏳ | CUDA backend, профили, бенчи, golden-тесты |

## Вехи (Definition of Done)

- ✅ **Ф0 DoD:** Workspace структура, CI, базовые трейты и типы
- ✅ **Ф1 DoD:** CLI сухой прогон — нормализованный текст → токены (dry-run)
- ✅ **Ф2 DoD:** Генерация acoustic токенов, KV cache, sampling strategies
- ✅ **Ф3 DoD:** Декодер выдаёт аудио, WAV экспорт, чанки без щелчков
- ✅ **Ф4 DoD:** TtsPipeline, StreamingSession, BatchScheduler
- ✅ **Ф5 DoD:** CLI синтезирует текст в WAV файл и в stream-режиме
- ✅ **Ф6 DoD:** gRPC сервер стримит аудио, health/metrics endpoints
- ⏳ **Ф7 DoD:** GPU RTF < 0.2, golden-тесты, бенчи

## Текущий статус

**Все основные фазы (0-6) завершены.** Проект имеет:

- 8 crates в workspace
- 149+ unit и integration тестов
- Mock-реализация полного pipeline
- CLI для синтеза
- gRPC сервер с HTTP endpoints

**Следующие шаги (Ф7):**

1. Загрузка реальных весов модели Qwen3-TTS
2. CUDA backend через candle-cuda
3. Профилирование и оптимизация
4. Golden tests с эталонными аудио
5. Benchmark suite

## Зависимости и порядки

```
Ф0 → Ф1 → Ф2 → Ф3 → Ф4 → Ф5
              ↓       ↓
            Ф4  →   Ф6
                      ↓
                    Ф7
```

## Риски

- Несовместимость токенизации с референсом → падение качества
- Неполный декодер 12Hz → артефакты/щелчки
- KV-cache память на GPU при батчинге → OOM
- CUDA/драйвер несостыковки; нужен fallback на CPU
- Латентность первого чанка > целевой — требуется микро-батч/прогрев

## Артефакты по фазам

| Фаза | Артефакты |
|------|-----------|
| Ф0 | Workspace, Cargo.toml, CI templates, базовые типы |
| Ф1 | Normalizer (6 правил), MockTokenizer, 31 golden test |
| Ф2 | Transformer layers, KV cache, Sampler, Model loader |
| Ф3 | NeuralDecoder, MockDecoder, Crossfader, WAV I/O |
| Ф4 | TtsPipeline, StreamingSession, BatchScheduler, TtsRuntime |
| Ф5 | tts-cli binary, synth/normalize/tokenize команды |
| Ф6 | tts-server binary, proto/tts.proto, gRPC + HTTP endpoints |
| Ф7 | Benchmarks, profiles, golden audio, stress reports |
