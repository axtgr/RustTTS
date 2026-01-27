# RustTTS vs Python SDK Benchmark

Сравнение производительности Rust реализации с официальным Python SDK Qwen3-TTS.

## Тестовая конфигурация

| Параметр | Значение |
|----------|----------|
| Hardware | Apple Silicon (M-серия) |
| OS | macOS |
| Python | 3.10 + PyTorch MPS (float32) |
| Rust | Candle 0.8 + CPU (GGUF Q8) |
| **Модель** | **Qwen3-TTS 0.6B CustomVoice (GGUF Q8)** (одинаковая для обоих) |
| **Текстовые примеры** | **Одинаковые для обоих (7, 73, 163 символов)** |

## Результаты

### Python SDK (MPS/GPU)

| Текст | Символов | Аудио (сек) | Синтез (сек) | RTF |
|-------|----------|-------------|--------------|-----|
| short ("Привет!") | 7 | 1.52 | 3.94 | **2.59x** |
| medium | 73 | 4.30 | 9.83 | **2.29x** |
| long | 163 | 11.92 | 23.20 | **1.95x** |

**Модель:** Qwen3-TTS-12Hz-0.6B-CustomVoice  
**Устройство:** MPS (Metal GPU, Apple Silicon)  
**Загрузка модели:** 7.7 сек  
**Память модели:** ~2 GB

### RustTTS (CPU)

| Текст | Символов | Аудио (сек) | Синтез (сек) | RTF |
|-------|----------|-------------|--------------|-----|
| short ("Привет!") | 7 | 1.50 | 4.85 | **3.24x** |
| medium | 73 | 5.26 | 7.52 | **1.43x** |
| long | 163 | 12.86 | 17.62 | **1.37x** |

**Модель:** qwen3-tts-0.6b-customvoice (GGUF Q8)  
**Устройство:** CPU  
**Загрузка модели:** ~1.92 сек  
**Память модели:** ~1.5 GB

### RustTTS (Metal)

| Текст | Символов | Аудио (сек) | Синтез (сек) | RTF |
|-------|----------|-------------|--------------|-----|
| short ("Привет!") | 7 | 1.58 | 8.35 | **5.30x** |
| medium | 73 | 4.22 | 14.67 | **3.48x** |
| long | 163 | 13.66 | 44.95 | **3.29x** |

**Модель:** qwen3-tts-0.6b-customvoice (GGUF Q8)  
**Устройство:** Metal  
**Загрузка модели:** ~2.20 сек  
**Память модели:** ~1.5 GB

## Сводная таблица (RTF - чем меньше, тем быстрее)

| Метрика | Python SDK (MPS GPU) | RustTTS (CPU) | RustTTS (Metal) | Победитель |
|---------|---------------------|---------------|-----------------|------------|
| Short RTF | **2.59x** | 3.24x | 5.30x | Python |
| Medium RTF | 2.29x | **1.43x** | 3.48x | Rust |
| Long RTF | 1.95x | **1.37x** | 3.29x | Rust |
| Cold start | ~7.7s | **~1.92s** | ~2.20s | Rust |
| Размер | ~2 GB (venv) | **~12 MB** | ~12 MB | Rust |
| Потребление RAM | ~2 GB | **~1.5 GB** | ~1.5 GB | Rust |

## Анализ

### Почему Python быстрее?

1. **MPS GPU ускорение** — Python SDK использует Metal Performance Shaders на Apple Silicon
2. **Оптимизированный PyTorch backend** — годы оптимизации под GPU inference
3. **Batch operations** — эффективные матричные операции на GPU

### Преимущества RustTTS

1. **Нет Python runtime** — чистый бинарник без интерпретатора
2. **Минимальные зависимости** — ~12MB binary vs ~2GB Python environment
3. **Быстрый cold start** — ~1.3 сек vs ~7-10 сек для Python
4. **Предсказуемость** — нет GC пауз, детерминированное поведение
5. **Легкое развертывание** — один файл без окружения

### Когда выбрать RustTTS

- Edge deployment (устройства без Python)
- Контейнеры с жесткими лимитами на размер
- Системы с требованиями к предсказуемому latency
- Embedded системы и IoT

### Когда выбрать Python SDK

- Максимальная производительность на GPU (MPS/CUDA)
- Прототипирование и эксперименты
- Уже есть Python инфраструктура
- Доступ к GPU (Apple Silicon / NVIDIA)

## План оптимизации RustTTS

1. **Metal GPU support** — добавить MPS бэкенд в Candle (частично готово, нужна доработка кернелов)
2. **Quantization** — INT8/INT4 для уменьшения памяти и ускорения
3. **SIMD оптимизации** — для CPU inference
4. **Streaming optimization** — уменьшить overhead

## Известные проблемы

### EOS Early Termination

Модель иногда генерирует EOS токен слишком рано. Применен workaround с минимальным количеством токенов на основе длины текста:

```rust
// pipeline.rs
let estimated_duration_s = (text_tokens.len() as f32 * 0.1).max(0.5);
let min_tokens = (estimated_duration_s * 12.0) as usize;
```

**TODO:** Исследовать root cause этой проблемы.

## Тестовые тексты

```
short:  "Привет!" (7 символов)
medium: "Ниже представлен план цикла статей о том, как создать собственную модель." (73 символа)
long:   "Ниже представлен план цикла статей о том, как создать собственную модель 
         преобразования текста в речь. Мы рассмотрим архитектуру, обучение и 
         развертывание системы." (163 символа)
```

## Как воспроизвести

### Python SDK

```bash
source /tmp/benchmark_venv310/bin/activate
python /tmp/python_benchmark.py
```

### RustTTS

```bash
cd /Users/askid/Projects/RustTTS
cargo build --release

# Build benchmark script
chmod +x /tmp/run_rust_benchmark.sh
/tmp/run_rust_benchmark.sh
```

---

*Benchmark date: 2026-01-28*  
*Model: Qwen3-TTS-12Hz-0.6B-CustomVoice*
