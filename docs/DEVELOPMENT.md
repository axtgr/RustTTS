# Руководство разработчика (Development Guide)

Этот документ описывает, как настроить окружение, собрать проект и запустить тесты.

## Предварительные требования (Prerequisites)

Для работы над проектом вам понадобятся:

1.  **Rust Toolchain**: Стабильная версия 1.75+
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2.  **Системные зависимости**:
    - `build-essential` / `gcc` (для сборки крейтов)
    - `pkg-config`, `libssl-dev` (стандартный набор)
    - `protobuf-compiler` (для gRPC сервера)

3.  **CUDA Toolkit** (Опционально, для GPU backend):
    - Рекомендуемая версия: 11.8 или 12.x
    - Убедитесь, что `nvcc` доступен в PATH

> **Важно:** Python запрещен в runtime кода, но может использоваться разработчиком локально для скачивания весов или конвертации форматов.

## Структура Workspace

Проект организован как Cargo Workspace из 8 крейтов:

```text
RustTTS/
├── Cargo.toml              # Workspace config
├── README.md               # Обзор проекта
├── AGENTS.md               # Инструкции для AI агентов
├── docs/                   # Документация
│   ├── architecture.md     # Архитектура системы
│   ├── roadmap.md          # Дорожная карта
│   ├── DEVELOPMENT.md      # Это руководство
│   └── phases/             # Документация по фазам
├── proto/                  # gRPC proto-файлы
│   └── tts.proto           # Определения сервисов TTS
├── crates/
│   ├── tts-core/           # Общие трейты и типы
│   ├── text-normalizer/    # Нормализация текста (числа, даты, валюты)
│   ├── text-tokenizer/     # BPE токенизация (HuggingFace tokenizers)
│   ├── acoustic-model/     # Transformer inference (candle)
│   ├── audio-codec-12hz/   # Декодер 12Hz → PCM, streaming
│   ├── runtime/            # Оркестратор, батчинг, пайплайн
│   ├── tts-cli/            # CLI утилита
│   └── tts-server/         # gRPC + HTTP сервер
└── models/                 # Локальная папка для весов (git-ignored)
```

## Сборка

### Полная сборка workspace
```bash
cargo build --workspace --release
```

### Сборка конкретного крейта
```bash
cargo build -p tts-core
cargo build -p acoustic-model --features cuda
cargo build -p tts-server --features server
```

### Быстрая проверка (без генерации бинарников)
```bash
cargo check --workspace
```

## Запуск

### CLI
```bash
# Синтез речи в WAV файл
cargo run -p tts-cli --release -- synth "Привет, мир!" -o output.wav

# Streaming режим
cargo run -p tts-cli --release -- synth "Длинный текст..." -o out.wav --streaming

# С указанием модели
cargo run -p tts-cli --release -- synth "Текст" -o out.wav --model ./models/qwen3-tts
```

### Сервер
```bash
# Запуск gRPC сервера (порт 50051) и HTTP (порт 8080)
cargo run -p tts-server --release

# С кастомным конфигом
cargo run -p tts-server --release -- --config server.toml
```

### Проверка сервера
```bash
# HTTP health check
curl http://localhost:8080/health

# Метрики
curl http://localhost:8080/metrics

# Информация о сервере
curl http://localhost:8080/info
```

## Тестирование

### Запуск всех тестов
```bash
cargo test --workspace
```

### Тесты конкретного крейта
```bash
cargo test -p tts-core
cargo test -p text-normalizer
cargo test -p text-tokenizer
cargo test -p acoustic-model
cargo test -p audio-codec-12hz
cargo test -p runtime
```

### Запуск конкретного теста
```bash
cargo test test_normalize_numbers
cargo test -p text-normalizer test_currency
```

### Тесты с выводом
```bash
cargo test -- --nocapture
```

### Golden тесты (требуют модельных весов)
```bash
cargo test --features golden_tests
```

### Игнорируемые/тяжёлые тесты
```bash
cargo test -- --ignored
```

## Линтинг и форматирование

**Обязательно перед каждым коммитом:**

```bash
# Форматирование кода
cargo fmt --all

# Проверка форматирования (CI)
cargo fmt --all -- --check

# Clippy линты (должны проходить без warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Clippy со всеми фичами
cargo clippy --workspace --all-features -- -D warnings
```

## Бенчмарки

```bash
# Все бенчмарки
cargo bench

# Конкретный бенчмарк
cargo bench --bench latency
cargo bench -p acoustic-model
```

## Документация кода

```bash
# Генерация и открытие документации
cargo doc --open --no-deps

# Документация всего workspace
cargo doc --workspace --no-deps
```

## Feature Flags

| Feature | Описание | Крейты |
|---------|----------|--------|
| `default` | CPU backend | все |
| `cuda` | GPU backend через candle-cuda | acoustic-model |
| `server` | gRPC сервер | tts-server |
| `golden_tests` | Golden тесты с эталонами | все |
| `bench` | Бенчмарки | все |

### Примеры использования
```bash
# Сборка с CUDA
cargo build -p acoustic-model --features cuda

# Запуск golden тестов
cargo test --features golden_tests

# Полная сборка со всеми фичами
cargo build --workspace --all-features
```

## CI/CD

Проект использует GitHub Actions. Пайплайн включает:

1. **cargo fmt** - проверка форматирования
2. **cargo clippy** - статический анализ
3. **cargo test** - юнит и интеграционные тесты
4. **cargo build --release** - сборка релиза

Конфигурация: `.github/workflows/ci.yml`

## Архитектура тестов

| Уровень | Описание | Запуск |
|---------|----------|--------|
| Unit | Быстрые тесты внутри модулей | `cargo test` |
| Integration | Тесты между крейтами | `cargo test -p runtime` |
| Golden | Сравнение с эталонами | `cargo test --features golden_tests` |
| Benchmark | Замеры производительности | `cargo bench` |

## Полезные команды

```bash
# Очистка артефактов сборки
cargo clean

# Обновление зависимостей
cargo update

# Проверка устаревших зависимостей
cargo outdated

# Дерево зависимостей
cargo tree

# Аудит безопасности
cargo audit

# Размер бинарников
cargo bloat --release -p tts-cli
```

## Отладка

```bash
# Запуск с debug логами
RUST_LOG=debug cargo run -p tts-cli -- synth "Test" -o out.wav

# Трассировка
RUST_LOG=trace cargo run -p tts-server

# Backtrace при панике
RUST_BACKTRACE=1 cargo test
```

## Troubleshooting

### Ошибка сборки protobuf
```bash
# macOS
brew install protobuf

# Ubuntu
sudo apt install protobuf-compiler
```

### CUDA не найден
```bash
# Проверьте PATH
which nvcc

# Установите переменные
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Тесты падают с OOM
```bash
# Ограничьте параллелизм
cargo test -j 4
```
