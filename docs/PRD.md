# PRD — Qwen3-TTS Rust Engine (без Python)

## Область и цель
Создать полноценную TTS-платформу на Rust, совместимую по поведению с Qwen3-TTS, с минимальной задержкой, поддержкой streaming, CUDA backend, батчингом и QoS. Никаких зависимостей на Python/torch.

## Основные требования
- Языки: RU/EN, смешанные фразы.
- Пайплайн: нормализация → токенизация → трансформер (акустические токены) → декодер Qwen3-TTS-Tokenizer-12Hz → PCM.
- Streaming: чанки 20–100 мс, overlap/crossfade, tail padding, управление буфером.
- Производительность: t2fa GPU 150–400 мс, CPU 600–1500 мс; RTF GPU < 0.2.
- Совместимость: веса в `safetensors`; конфиги моделей; deterministic режим.
- Эксплуатация: structured logs, метрики, бенчи, unit + golden tests.
- Интерфейсы: `tts-cli`; опционально `tts-server` (HTTP/2 или gRPC streaming).

## Крейты и роли
- `tts-core`: общие трейты, типы, ошибки, `Synthesizer` API.
- `text-normalizer`: правила RU/EN, плагинный пайплайн.
- `text-tokenizer`: совместимый BPE/Unigram, потоковый encode, offsets.
- `acoustic-model`: трансформер с KV cache, streaming генерация.
- `audio-codec-12hz`: декодер токенов → PCM, overlap-add/crossfade.
- `runtime`: очереди, батчинг, QoS, метрики/логи, device mgmt.
- `tts-cli`: оффлайн/stream CLI, бенчи/dry-run.
- `tts-server` (опц.): streaming API, health/metrics, auth.

## Ссылки на детали
- Roadmap: `docs/roadmap.md`.
- Архитектура и API: `docs/architecture.md`.
- Фазы и задачи: `docs/phases/phase0.md` … `phase6.md`.

## Ключевые архитектурные решения
- Tensor backend: приоритет `candle` (CUDA/CPU), абстракция `TensorBackend` для замены.
- KV cache: device-resident, кольцевой буфер, LRU-эвикция на уровне сессий.
- Батчинг: микро-окно 5–20 мс, приоритет stream, лимиты памяти/tokens.
- Streaming аудио: выдача чанков 20–100 мс, overlap-add Hann 5–10 мс.
- Конфиги: `model.toml`, `runtime.toml`, `server.toml`; версия весов = версия кода.
- Логи/метрики: `tracing` JSON, Prometheus метрики (latency/RTF/drop/память/GPU util).

## Качество и тестирование
- Unit: нормализация, токенизация, KV cache, crossfade, PCM конверсия.
- Golden: текст → токены/PCM; метрики SDR/MSE; версии зафиксированы.
- Бенчи: t2fa, RTF (CPU/GPU), память, cold/warm start; мок-бенчи в CI.
- Стресс: 10+ минут стрим, N параллельных запросов, QoS (дедлайны/отмены).

## Риски и смягчение
- Несовместимость токенов/кодека: golden + checksum vocabs/весов.
- CUDA/драйверы: матрица версий, fallback CPU.
- KV cache память: лимиты, эвикция, профили.
- Латентность t2fa: прогрев, микро-батч, стримовая токенизация, ранний decode.

## Definition of Success
- End-to-end стрим с p95 t2fa в целевых пределах, RTF GPU < 0.2.
- Отсутствие щелчков/швов на чанках; метрики/логи доступны; тесты/голдены проходят.
