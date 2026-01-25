# Фаза 6 — Оптимизация и качество (CUDA, бенчи, golden, стресс)

## Цель
Достичь целевых KPI (t2fa, RTF) на GPU/CPU, убедиться в стабильности и регрессии качества.

## Epics → Stories → Subtasks
- Epic: CUDA backend
  - Story: Сборка
    - Subtask: Флаги/сборка candle/tch с CUDA.
    - Subtask: Проверка совместимости драйверов/версий.
  - Story: Оптимизации
    - Subtask: Fused attention (если доступно), fp16/bf16, опц. int8.
    - Subtask: KV cache на GPU, профили памяти.
- Epic: Бенчмарки и профили
  - Story: Бенчи
    - Subtask: t2fa (cold/warm), RTF по длинам текстов.
    - Subtask: GPU util, память.
  - Story: Профилирование
    - Subtask: Flamegraph CPU; NVTX/NSight метки для GPU (если возможно).
- Epic: Качество (golden)
  - Story: Golden аудио
    - Subtask: Сравнение SDR/MSE с эталоном.
    - Subtask: Перцептивная проверка (manually curated список).
- Epic: Стресс/надёжность
  - Story: Долгие стримы 10+ минут
    - Subtask: Мониторинг утечек памяти/дескрипторов.
  - Story: Нагрузка N параллельных стримов
    - Subtask: Проверка QoS (дроп/тайм-ауты, очереди).

## Deliverables
- CUDA backend включаемый флагом, документация по установке.
- Бенч-отчёты (таблицы t2fa/RTF), профили.
- Golden-тесты и допуски.
- Стресс-отчёт (стабильность, drop rate).

## DoD
- GPU RTF < 0.2 на реф. модели/железе.
- CPU RTF в заявленных пределах; без утечек на 10+ минут стрима.
- Golden-тесты проходят в допусках; бенчи опубликованы.

## Риски
- Несовместимость CUDA окружений; смягчение — документировать матрицу версий, fallback CPU.
- Fused attention/FP16 могут давать расхождения; смягчение — режим deterministic для тестов.

## Метрики проверки
- p95 t2fa, RTF GPU/CPU на реф. корпусе.
- Память: отсутствие роста на длительном стриме.
- Drop/timeout rate в стресс-тесте ≤ целевого (стремиться к 0).
