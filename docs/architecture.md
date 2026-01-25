# Архитектура Qwen3-TTS Rust Engine

## Общий обзор
- Workspace из нескольких крейтов: `tts-core`, `text-normalizer`, `text-tokenizer`, `acoustic-model`, `audio-codec-12hz`, `runtime`, `tts-cli`, опц. `tts-server`.
- Конвейер: normalize → tokenize → acoustic tokens → decode to PCM → чанки с overlap/crossfade → вывод (CLI/Server).
- Цели: минимальная latency до первого аудио, RTF<0.2 на GPU, streaming 20–100 мс.

## Референсы и источники
- **Qwen2-Audio / Qwen-TTS Paper**: Основной источник архитектуры модели и математики.
- **Original Repository**: Ссылка на оригинальный Python-код (использовать только для сверки тензорных операций и логики).
- **Candle / Tch**: Документация по Rust ML-фреймворкам.

## Ключевые решения
- Бэкенд тензоров: приоритет `candle` (Rust-first, CUDA), fallback CPU. Рассмотреть `tch` если нужны成熟ые CUDA фичи (flash-attn), оставить абстракцию через trait `TensorBackend`.
- KV cache: храним на выбранном device; кольцевой буфер по токенам, ключ `SessionId+Layer`; eviction LRU на уровне батчера.
- Батчинг: микро-окно 5–20 мс; приоритетные (stream) запросы выше batch; лимит по токенам/памяти.
- Streaming: генерация шагами, немедленный декод в PCM чанки 20–100 мс; overlap-add (Hann окно 5–10 мс).
- Конфиги: `model.toml`/`config.json` с dims, vocab, sample_rate, codec_stride, пути весов (safetensors), режим device.
- Логи/метрики: `tracing` JSON, `metrics` + Prometheus exporter; кореляция через trace_id.

## Модули и интерфейсы

### tts-core
- Общие типы и трейты: `TextNormalizer`, `TextTokenizer`, `AcousticModel`, `AudioCodec`, `Synthesizer`.
- Типы данных: `NormText`, `TokenSeq`, `AcousticStep`, `AudioChunk`, `SynthesisRequest`, `BatchItem`.
- Ошибки/результаты: единый `TtsError`, контекст, категории (IO/Model/Decode/Timeout/QoS).

### text-normalizer
- Пайплайн правил (trait Rule): последовательное применение; поддержка RU/EN, смешанных текстов.
- Обработка чисел, дат, валют, единиц; унификация символов; безопасные падения (не ломать исходный текст).
- Тестовые корпуса + golden.

### text-tokenizer
- Совместимый BPE/Unigram (с моделью Qwen3); BOS/EOS/паузные/служебные токены.
- Потоковый encode (по предложениям) для низкой latency; decode для отладки.
- Детализация offsets для обратной связи/логов.

### acoustic-model
- Трансформер inference: эмбеддинги, attention с KV cache, rotary, MLP.
- Streaming автогрессия: выдача токенов шагами; greedy/top-k/top-p/temperature; опц. beam search.
- Backends: CPU (SIMD/rayon), GPU (CUDA через candle/tch). Режимы fp16/bf16, опц. int8.
- Загрузка весов `safetensors`, ленивое mmap; валидация версии конфигов.

### audio-codec-12hz
- Декодер Qwen3-TTS-Tokenizer-12Hz: токены → PCM (f32 → i16 LE / WAV).
- Потоковый API: `next_chunk` выдаёт 20–100 мс; поддержка overlap-add/crossfade.
- Опц. обратный энкодер для тестов (валидировать round-trip на мини-кейсах).

### runtime
- Оркестратор: очереди запросов, батчер, QoS политики (приоритет, deadline, отмена).
- Менеджер device: CPU/GPU выбор, прогрев, профили (low-latency/high-throughput).
- Буферизация аудио: предотвращение underrun/overrun, плавное завершение (tail padding).
- Structured logging + метрики; экспорт `/metrics` (если сервер включён).

### tts-cli
- Режимы: файл → WAV; stdin → stream PCM stdout; параметры: язык, speaker_id, chunk_ms, device, fp16, seed.
- Диагностика: `--bench` (локальный бенч), `--dry-run` (токены без декода).

### tts-server (опционально)
- gRPC или HTTP/2 streaming; endpoints: `/synthesize` (stream), `/health`, `/metrics`.
- Auth токен; лимиты запросов; graceful shutdown; backpressure.

## Структуры данных (примерно)
- `NormText { text: String, lang: Lang, spans: Vec<SpanInfo> }`
- `TokenSeq { ids: Vec<u32>, offsets: Vec<usize> }`
- `AcousticStep { token: u32, prob: f32, t_ms: f32 }`
- `AudioChunk { pcm: Arc<[f32]>, sample_rate: u32, start_ms: f32, end_ms: f32 }`
- `SynthesisRequest { text, lang, speaker_id, chunk_ms, max_latency_ms, priority, session_id, seed }`
- `BatchItem { req_id, tokens_in, kv_cache_handle, deadline, priority }`

## API (сигнатуры на уровне трейтов)
- `TextNormalizer::normalize(&self, input: &str, lang_hint: Option<Lang>) -> NormText`
- `TextTokenizer::encode(&self, NormText) -> TokenSeq`
- `AcousticModel::generate_stream(&self, tokens: TokenSeq, opts: GenOpts) -> impl Stream<Item=AcousticStep>`
- `AudioCodec::decode_stream(&self, steps: impl Stream<Item=AcousticStep>, opts: DecodeOpts) -> impl Stream<Item=AudioChunk>`
- `Synthesizer::synthesize_stream(&self, req: SynthesisRequest) -> impl Stream<Item=AudioChunk>`

## Конфиги
- `model.toml`: dims (hidden, heads, layers), vocab, codec params (stride, frame_hop, sample_rate), backend prefs, quantization, seed.
- `runtime.toml`: очереди, батч-окно, max batch, QoS приоритеты, время ожидания, лог-уровни.
- `server.toml`: адрес, лимиты, auth, метрики, CORS (если HTTP).

## Логирование и метрики
- `tracing` JSON; поля: trace_id, session_id, req_id, stage, latency_ms, chunk_idx, batch_size.
- Метрики: p50/p95 latency, time-to-first-audio, RTF, GPU util (если доступно), drop/timeout rate, memory.

## Качество и тесты
- Unit: правила нормализации, токенизация, KV cache индекс, crossfade.
- Golden: текст → эталонные токены/PCM (MSE/SDR допуск), версии фиксировать.
- Бенч: cold start, t2fa, RTF по длине текста, память; профили CPU/GPU.
- Нагрузочные: многопоточные запросы, QoS (дедлайны, отмены, drops), долгий стрим 10+ мин.

## Риски
- Несовпадение токенов с эталоном; смягчение: golden corpus + регрессия.
- Некорректный декодер 12Hz; смягчение: сверка спектров/SDR, обратный энкодер для теста.
- OOM GPU из-за KV cache; смягчение: лимиты, эвикция, профили памяти.
- Латентность первого чанка; смягчение: прогрев, микро-батч, стримовая токенизация.
- CUDA совместимость; смягчение: тест-матрица версий, CPU fallback.
