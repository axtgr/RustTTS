# Фаза 2 — Акустическая модель (трансформер, KV cache, streaming)

## Цель
Реализовать генерацию акустических токенов на CPU (baseline) и подготовить к CUDA, с поддержкой streaming и KV cache.

## Epics → Stories → Subtasks
- Epic: Загрузка модели
  - Story: Safetensors loader
    - Subtask: Парсинг метаданных, сверка конфигов.
    - Subtask: Ленивое mmap, проверка checksum.
  - Story: Конфигурация трансформера
    - Subtask: Валидация dims (hidden, heads, layers, rotary params).
    - Subtask: Спец-токены остановки/пауз.
- Epic: Инференс движок
  - Story: CPU backend
    - Subtask: Матмул/attention через candle CPU (или SIMD + rayon).
    - Subtask: KV cache layout (layer, head, seq_pos) + кольцевой буфер.
  - Story: Streaming автогрессия
    - Subtask: Генерация step-by-step; управление max_tokens, stop_tokens.
    - Subtask: Sampling: greedy, top-k, top-p, temperature; опц. beam.
  - Story: API
    - Subtask: `generate_stream` возвращает stream `AcousticStep` с временными метками.
- Epic: Тесты и валидация
  - Story: Unit
    - Subtask: KV cache индексация и обнуление.
    - Subtask: Sampling корректность (стохастические тесты с seed).
  - Story: Golden (мини-модель)
    - Subtask: Входные токены → эталонные выходные токены (фиксированный seed).
  - Story: Бенч
    - Subtask: Латентность t2fa (tok→first acoustic) на CPU.
    - Subtask: RTF CPU baseline.

## Deliverables
- Крейты `acoustic-model` с CPU backend и streaming API.
- Мини-веса для тестов/golden.
- KV cache реализация и документация формата.

## DoD
- Генерация акустических токенов стабильно, deterministic при фиксированном seed.
- Streaming API выдаёт токены шагами без утечек памяти.

## Риски
- Ошибки в KV cache приводят к деградации качества; смягчение — тесты на reset/offsetы.
- Медленный CPU путь → RTF > целевого; смягчение — профилировка, SIMD, батч.

## Метрики проверки
- Time-to-first-acoustic (CPU p95) < 150 мс на реф. тексте.
- RTF CPU (baseline) <= 2.0 на реф. машине.
