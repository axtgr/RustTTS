# Фаза 3 — Аудио-декодер Qwen3-TTS-Tokenizer-12Hz → PCM

## Цель
Декодировать акустические токены в PCM с низкой задержкой, обеспечивая streaming чанки 20–100 мс с overlap/crossfade.

## Epics → Stories → Subtasks
- Epic: Декодер 12Hz
  - Story: Загрузка весов
    - Subtask: Safetensors loader для кодека; проверка stride/sample_rate.
  - Story: Инференс
    - Subtask: Декодер токенов → PCM (f32), затем i16/WAV.
    - Subtask: Поддержка батча фреймов.
- Epic: Streaming вывод
  - Story: Чанки
    - Subtask: Размеры 20/40/60/100 мс настраиваемо.
    - Subtask: Tail padding при завершении.
  - Story: Overlap/Crossfade
    - Subtask: Хранение хвоста N сэмплов.
    - Subtask: Окно Hann (5–10 мс) для склейки.
- Epic: Тесты и качество
  - Story: Unit
    - Subtask: Корректность окна и индексов.
    - Subtask: Конвертация PCM f32 → i16 LE.
  - Story: Golden
    - Subtask: Токены → эталонный WAV (MSE/SDR допуск).
  - Story: Бенчи
    - Subtask: Латентность токены→PCM для коротких/длинных фраз.

## Deliverables
- Крейты `audio-codec-12hz` с потоковым API `next_chunk`.
- Реализация overlap-add/crossfade.
- Тесты (unit + golden) и пример WAV.

## DoD
- Потоковые чанки без щелчков/швов; SDR/MSE в пределах допуска на golden.
- Конвертация в WAV/i16 корректна; tail закрывает стрим без артефактов.

## Риски
- Неполная реализация кодека → артефакты; смягчение — сверка спектра и SDR.
- Ошибки в кроссфейде → щелчки; смягчение — unit на окна, прослушивание.

## Метрики проверки
- MSE/SDR против эталона в допуске.
- Латентность токены→первый PCM chunk < 50 мс (GPU), < 120 мс (CPU реф.).
