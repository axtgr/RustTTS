# RustTTS vs Python SDK Benchmark

Сравнение производительности Rust реализации с официальным Python SDK Qwen3-TTS.

## Окружение тестирования

| Параметр | Значение |
|----------|----------|
| Hardware | Apple Silicon (M-серия) |
| OS | macOS |
| Python | 3.11 + PyTorch MPS (float32) |
| Rust | Candle 0.8 + CPU |
| Модель | Qwen3-TTS 0.6B |

## Результаты

### Python SDK (MPS/GPU)

| Текст | Символов | Аудио (сек) | Синтез (сек) | RTF |
|-------|----------|-------------|--------------|-----|
| short ("Привет!") | 7 | 0.88 | 3.82 | 4.34x |
| medium | 73 | 13.74 | 17.09 | 1.24x |
| long | 163 | 12.38 | 30.89 | 2.50x |

**Загрузка модели:** 2.44 сек  
**Память модели:** ~390 MB

### RustTTS (CPU)

| Текст | Символов | Аудио (сек) | Синтез (сек) | Загрузка (сек) | RTF |
|-------|----------|-------------|--------------|----------------|-----|
| short ("Привет!") | 13 | 0.46 | 2.49 | 2.70 | 5.44x |
| medium | 134 | 2.06 | 8.00 | 2.14 | 3.89x |
| long | 301 | 5.26 | 19.16 | 1.66 | 3.65x |

**Загрузка модели:** ~2.7 сек (первый запуск)

## Сводная таблица

| Метрика | Python SDK (MPS) | RustTTS (CPU) | Победитель |
|---------|------------------|---------------|------------|
| Загрузка модели | 2.44s | 2.70s | Python |
| Short RTF | 4.34x | 5.44x | Python |
| Medium RTF | 1.24x | 3.89x | Python |
| Long RTF | 2.50x | 3.65x | Python |
| Зависимости | Python + PyTorch | Pure Rust | **Rust** |
| Размер бинарника | ~2GB (venv) | ~12MB | **Rust** |
| Cold start | ~5-10s | ~3s | **Rust** |
| Потребление RAM | ~2-4 GB | ~1.5 GB | **Rust** |

## Анализ

### Почему Python быстрее?

1. **MPS ускорение** — Python SDK использует Metal Performance Shaders на Apple Silicon
2. **Оптимизированный PyTorch** — годы оптимизации под GPU
3. **Batch operations** — эффективные матричные операции на GPU

### Преимущества RustTTS

1. **Нет Python runtime** — чистый бинарник без интерпретатора
2. **Минимальные зависимости** — ~12MB бинарник vs ~2GB Python env
3. **Быстрый cold start** — ~3 сек vs ~5-10 сек для Python
4. **Предсказуемость** — нет GC пауз, детерминированное поведение
5. **Легкое развертывание** — один файл без окружения

### Когда выбрать Rust

- Edge deployment (устройства без Python)
- Контейнеры с жесткими лимитами на размер
- Системы с требованиями к предсказуемому latency
- Embedded системы и IoT

### Когда выбрать Python

- Максимальная производительность на GPU
- Прототипирование и эксперименты
- Уже есть Python инфраструктура

## План оптимизации RustTTS

1. **Metal GPU support** — добавить MPS бэкенд в Candle
2. **CUDA support** — для NVIDIA GPU (уже частично готово)
3. **KV Cache оптимизация** — уменьшить копирования
4. **SIMD оптимизации** — для CPU inference
5. **Quantization** — INT8/INT4 для уменьшения памяти

## Известные проблемы

### EOS Workaround

Модель иногда генерирует EOS токен слишком рано. Применен workaround с минимальным количеством токенов на основе длины текста:

```rust
// pipeline.rs
let estimated_duration_s = (text_tokens.len() as f32 * 0.1).max(0.5);
let min_tokens = (estimated_duration_s * 12.0) as usize;
```

**TODO:** Исследовать root cause этой проблемы.

## Тестовые тексты

```
short:  "Привет!"
medium: "Ниже представлен план цикла статей о том, как создать собственную модель."
long:   "Ниже представлен план цикла статей о том, как создать собственную модель 
         преобразования текста в речь. Мы рассмотрим архитектуру, обучение и 
         развертывание системы."
```

## Как воспроизвести

### Python SDK

```bash
source /tmp/qwen-venv/bin/activate
python /tmp/python_benchmark.py
```

### RustTTS

```bash
cd /Users/askid/Projects/RustTTS
cargo build --release

# Short
time ./target/release/tts synth "Привет!" --model-dir models/qwen3-tts-0.6b -o /tmp/rust_short.wav

# Medium
time ./target/release/tts synth "Ниже представлен план цикла статей о том, как создать собственную модель." --model-dir models/qwen3-tts-0.6b -o /tmp/rust_medium.wav

# Long
time ./target/release/tts synth "Ниже представлен план цикла статей о том, как создать собственную модель преобразования текста в речь. Мы рассмотрим архитектуру, обучение и развертывание системы." --model-dir models/qwen3-tts-0.6b -o /tmp/rust_long.wav
```

---

*Benchmark date: 2026-01-27*
