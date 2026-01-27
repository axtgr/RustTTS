# План реализации квантования (INT8/INT4) для Qwen3-TTS

## 1. Цели и мотивация
*   **Снижение потребления памяти:** Уменьшить размер модели в VRAM/RAM с ~2GB (FP16) до ~0.7-1.0GB (INT4/INT8).
*   **Ускорение на CPU:** Использование специализированных ядер для квантованных вычислений (AVX, NEON).
*   **Совместимость:** Поддержка формата GGUF и экосистемы Candle.

## 2. Анализ зависимостей
*   **candle-transformers:** Используется v0.8.
*   **candle-core:** Используется v0.8.
*   **Требование:** Убедиться, что фича `quantized` (или эквивалент для `candle-transformers`) включена в `Cargo.toml` для доступа к структурам `QMatMul`, `GgufFile`.

## 3. Архитектурные изменения

### 3.1. Абстракция линейного слоя (`crates/acoustic-model/src/layers.rs`)
Вместо прямой зависимости от `candle_nn::Linear`, вводим абстракцию:

```rust
use candle_transformers::quantized::QMatMul;

#[derive(Debug)]
pub enum LinearLayer {
    /// Стандартный линейный слой (F32/F16)
    Standard(candle_nn::Linear),
    /// Квантованный слой (GGUF: Q4_0, Q8_0, и т.д.)
    Quantized(QMatMul),
}

impl LinearLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(x),
            // QMatMul автоматически деквантует или использует оптимизированные ядра
            Self::Quantized(q) => q.forward(x), 
        }
    }
}
```

### 3.2. Обновление слоев трансформера
Заменить `candle_nn::Linear` на `LinearLayer` в структурах:
*   `Attention` (поля: `q_proj`, `k_proj`, `v_proj`, `o_proj`)
*   `MLP` (поля: `gate_proj`, `up_proj`, `down_proj`)
*   `TextProjection` (поля: `fc1`, `fc2`)
*   `Model` (поле: `codec_head`)

*Примечание:* Эмбеддинги (`Embedding`) оставляем стандартными или квантуем отдельно, если `candle` это поддерживает (обычно `QEmbeddings`), но приоритет — матрицы весов.

### 3.3. Логика загрузки (`crates/acoustic-model/src/model.rs`)
Обновить метод `Model::load`:
1.  Определять формат по расширению файла.
2.  Если `.safetensors` -> старая ветка (создаем `LinearLayer::Standard`).
3.  Если `.gguf`:
    *   Использовать `candle_transformers::quantized::gguf_file::Content::read`.
    *   Извлекать тензоры по именам.
    *   Создавать `LinearLayer::Quantized(QMatMul::from_arc(...))`.

## 4. Инструментарий конвертации

Создать CLI-утилиту для конвертации весов: `crates/tts-cli/src/bin/quantize.rs` (или подкоманду `tts-cli quantize`).

**Функционал:**
1.  Загрузка исходных весов (safetensors).
2.  Выбор типа квантования (по умолчанию `q8_0` для баланса, `q4_0` для макс. сжатия).
3.  Запись в формат GGUF с сохранением структуры имен тензоров (`talker.model.layers.0...`).

## 5. План работ (по шагам)

### Шаг 1: Рефакторинг слоев
*   Модифицировать `crates/acoustic-model/src/layers.rs`.
*   Внедрить `LinearLayer`.
*   Обеспечить компиляцию и прохождение тестов в режиме F32 (Standard).

### Шаг 2: Поддержка GGUF загрузки
*   Добавить зависимости в `Cargo.toml`.
*   Реализовать чтение GGUF в `crates/acoustic-model/src/model.rs`.
*   Написать тест загрузки (если есть тестовый GGUF, или заглушку).

### Шаг 3: Утилита конвертации
*   Реализовать команду `quantize`.
*   Проверить цикл: `safetensors` -> `quantize` -> `gguf` -> `load` -> `inference`.

### Шаг 4: Валидация
*   Сравнить метрики (аудио-выход) между FP32 и Q8_0.
*   Замерить потребление памяти.

## 6. Риски
*   Артефакты в аудио при INT4. **Mitigation:** Использовать Q8_0 как основной вариант.
*   Сложность маппинга имен в GGUF. **Mitigation:** Строгий нейминг при конвертации.
