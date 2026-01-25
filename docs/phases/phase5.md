# Фаза 5 — Интеграция CLI и сервера

## Цель
Предоставить удобные интерфейсы: CLI для оффлайн/стриминга и опциональный сервер со streaming API, health, metrics.

## Epics → Stories → Subtasks
- Epic: CLI (`tts-cli`)
  - Story: Режимы
    - Subtask: Файл → WAV.
    - Subtask: stdin → stream PCM на stdout (чанки).
  - Story: Опции
    - Subtask: Язык, voice/speaker_id, chunk_ms, device, fp16, seed.
    - Subtask: `--dry-run` (токены), `--bench` (локальный бенч t2fa/RTF).
- Epic: Сервер (`tts-server`, опц.)
  - Story: API
    - Subtask: HTTP/2 или gRPC streaming `/synthesize`.
    - Subtask: `/health`, `/metrics` (Prometheus).
  - Story: Безопасность/лимиты
    - Subtask: Auth token; лимиты запросов/соединений.
    - Subtask: Graceful shutdown.
- Epic: Документация и примеры
  - Story: README для CLI/Server, примеры команд и запросов.
  - Story: Контракты API (protobuf/OpenAPI кратко).

## Deliverables
- Бинарь `tts-cli`; опционально `tts-server`.
- Примеры использования и API-контракты.

## DoD
- CLI синтезирует текст в файл и в stream-режиме.
- Сервер (если включён) стримит PCM, возвращает health/metrics.

## Риски
- Стабильность stream при сетевых обрывах; смягчение — тайм-ауты, retry на клиенте (документация).
- Совместимость HTTP/2 vs gRPC; выбрать один приоритетно и задокументировать.

## Метрики проверки
- CLI: t2fa и RTF в целевых пределах на реф. железе.
- Сервер: p95 latency, error rate, корректность `/health`/`/metrics`.
