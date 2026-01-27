# RustTTS Performance Optimization Plan

**Status:** В разработке  
**Start Date:** 2026-01-27  
**Target:** Dogon and exceed Python SDK performance

## Problem Analysis

Tекущие результаты не конкурентоспособны:

| Метрика | Python (MPS GPU) | Rust (CPU) | Rust (Metal) | Победитель |
|---------|------------------|-----------|--------------|------------|
| Short RTF | 2.59x | 4.24x | 4.24x | Python ⭐ |
| Medium RTF | 2.29x | 2.81x | 2.81x | Python ⭐ |
| Long RTF | 1.95x | 3.33x | 3.33x | Python ⭐ |
| Cold start | ~8s | ~1.3s | ~3.5s | Rust ⭐ |
| Size | ~2GB | ~12MB | ~12MB | Rust ⭐ |
| RAM | ~2GB | ~1.5GB | ~1.5GB | Rust ⭐ |

### Root Causes

1. **Неоптимизированный Metal backend** - kernel launch overhead на каждом токене
2. **НЕТ Quantization** - использует f32 (в 4x медленнее INT8)
3. **НЕТ SIMD оптимизаций** - generic kernels вместо ARM NEON
4. **Architecture** - layer-by-layer вместо fused kernels

## Optimization Phases

### Phase 1: Metal Backend Fix (1-2 weeks) 🚨 СРОЧНО

#### 1.1 Profiler Integration
**Цель:** Найти bottleneck операции

```rust
// crates/runtime/src/profiler.rs
use tracing::{info, instrument};

#[instrument(skip(audio))]
pub fn profile_section(name: &str) -> impl Drop {
    let start = Instant::now();
    info!("Section {} started", name);
    
    struct ProfilerGuard {
        name: String,
        start: Instant,
    }
    
    impl Drop for ProfilerGuard {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            info!("{} elapsed: {:?}", self.name, elapsed);
        }
    }
    
    ProfilerGuard { name: name.into(), start }
}
```

**Использование:**
```rust
pub fn synthesize(...) -> Result {
    let _p = profile_section("normalization");
    // ... 
    
    let _p = profile_section("tokenization");  
    // ...
}
```

#### 1.2 Metal Kernel Optimization

**Проблемы:**
- Отдельный kernel для каждого токена autoregressive  
- НЕТ Fused Attention
- Generic kernels вместо оптимизированных

**Решения:**

**Option A: Fused Attention Kernel**
```cuda
// Metal shader
kernel void fused_attention(
    const device float* q,
    const device float* k,
    const device float* v,
    const device int* cache,
    device float* out,
    const uint seq_len
) {
    // Все операции attention в одном вызове
    // min memory transfers
}
```

**Option B: Batch Generation**
```rust
// Генерировать 2-4 токена за раз где возможно
// Уменьшить kernel launches в 2-4x
const BATCH_SIZE: usize = 4;
```

**Option C: Shader Caching**
```rust
// crates/runtime/src/metal_cache.rs
use std::collections::HashMap;
use std::sync::{Mutex, Arc};

struct ShaderCache {
    cache: Mutex<HashMap<String, CompiledMetalLibrary>>,
    
    fn get_or_compile(&self, source: &str) -> Result<CompiledMetalLibrary> {
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(shader) = cache.get(source) {
            return Ok(shader.clone());
        }
        
        let compiled = self.compile_shader(source)?;
        cache.insert(source.into(), compiled.clone());
        Ok(compiled)
    }
}
```

**Ожидаемый результат:** RTF 4.24x → 3.0x (~30% faster)

### Phase 2: Quantization (2-3 weeks) ⭐⭐⭐ ТОП ПРИОРИТЕТ

#### 2.1 INT8 Weights Storage
```toml
[dependencies]
candle-core = { version = "0.8", features = ["quantize"] }
candle-metal = { version = "0.8", features = ["int8"] }
```

```
Размер модели:
- f32: 2.5 GB
- INT8: ~700 MB (3.5x меньше!)
- INT4: ~350 MB (7x меньше!)
```

#### 2.2 INT8 MatMul Kernels

```rust
// Quantized matrix multiplication
pub fn matmul_int8_quantized(a: &[i8], b: &[i8], scale_a: f32, scale_b: f32) -> Vec<f32> {
    use std::arch::aarch64::vld1q_s8;
    use std::arch::aarch64::vdotq_s32;
    
    // SIMD optimized I8 matmul
    unsafe {
        for i in (0..n).step_by(16) {
            let va = vld1q_s8(a.as_ptr().offset(i));
            let vb = vld1q_s8(b.as_ptr().offset(i));
            let result = vdotq_s32(...);
            // ...
        }
    }
}
```

#### 2.3 INT8 Attention Ops

```rust
// Quantized self-attention
pub fn attention_int8(
    q: &[i8], k: &[i8], v: &[i8],
    scale_qk: f32, scale_v: f32
) -> Vec<f32> {
    // Сначала кэшировать INT8 dot product
    let scores = matmul_int8(q, k, scale_qk, 1.0);
    // Затем softmax + v matmul в INT8
    // Только на концах конвертация в f32
}
```

**Прогнозируемые результаты:**

| Backend | f32 сейчас | INT8 | INT4 |
|---------|-----------|------|------|
| CPU | 4.24x RTF | **2.5x RTF** | **2.0x RTF** |
| Metal | 4.24x RTF | **2.5x RTF** | **2.0x RTF** |
| vs Python | ❌ проигрывает | ✅ догнал! | ✅ на 30% быстрее! |

### Phase 3: SIMD Optimization (CPU) (2 weeks)

#### 3.1 ARM NEON for Apple Silicon

```rust
// crates/acoustic-model/src/ops/neon.rs
#![cfg(target_arch = "aarch64")]
#![feature(stdsimd)]

use std::arch::aarch64::*;

pub unsafe fn neon_matmul_f32x4(
    a: &[f32], b: &[f32], out: &mut [f32]
) {
    for i in (0..256).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().offset(i));
        let b_vec = vld1q_f32(b.as_ptr().offset(i));
        let result = vaddq_f32(a_vec, b_vec);
        vst1q_f32(out.as_mut_ptr().offset(i), result);
    }
}
```

#### 3.2 AVX512 for x86

```rust
// Для Linux servers
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub unsafe fn avx512_matmul(...) { /* ... */ }
```

**Ожидаемый результат:** 15-25% ускорение на CPU

### Phase 4: Architecture Optimizations (1-2 weeks)

#### 4.1 Fused Layer Kernels

**Сейчас:**
```rust
for layer in &model.layers {
    output = layer.attention(output, &mut cache)?;
    output = layer.mlp(output)?;
}
```

**Fused:**
```rust
// Один kernel для нескольких слоёв
output = model.fused_layers_1_2(output, &mut cache)?;
```

#### 4.2 KV Cache Ring Buffer

```rust
// crates/runtime/src/cache/ring.rs
pub struct RingBufferCache<T> {
    data: Vec<T>,
    head: usize,  // Newest
    tail: usize,  // Oldest
    capacity: usize,
}

impl<T: Clone> RingBufferCache<T> {
    const EVICT: usize = 64; // Batch eviction
    
    pub fn push(&mut self, item: T) {
        self.data[self.head] = item;
        self.head = (self.head + 1) % self.capacity;
        
        if self.head == self.tail {
            self.tail = (self.tail + Self::EVICT) % self.capacity;
        }
    }
    
    // Minimize allocations на 50%+
}
```

#### 4.3 Lazy Loading

```rust
// crates/runtime/src/loader.rs
pub struct LazyModelLoader {
    path: PathBuf,
    loaded: AtomicBool,
    model: OnceCell<Model>,
}

impl LazyModelLoader {
    pub fn get(&self) -> Result<&Model> {
        self.model.get_or_try_init(|| {
            Model::load(self.path)
        })
    }
}
```

**Ожидаемый результат:** 10-20% ускорение

### Phase 5: Real-World Metrics (1 week)

RTF не главное для TTS! Важно:

#### 5.1 Time-to-First-Audio (TTFA)
```
Цель: < 100ms от запроса до первого фрагмента аудио

Методика:
- Отметить t₀ = request received
- Отметить t₁ = first audio chunk ready => TTFA = t₁ - t₀
- Python: ~500ms (требует вся генерация завершить)
- Rust цель: <100ms (streaming!)
```

#### 5.2 Streaming Throughput
```rust
// chunks каждые 20-50ms во время генерации
// не дожидаться полной генерации (как делает Python)
pub fn generate_streaming(text: &str) -> impl Stream<Item = AudioChunk> {
    // Stream generator yields сразу как есть данные
    let mut pos = 0;
    loop {
        let chunk = generate_next_chunk(&text[pos..pos + CHUNK_SIZE])?;
        pos += chunk.duration;
        if pos >= text.len() { break; }
        yield chunk;
    }
}
```

#### 5.3 Memory Footprint
```rust
// Peak memory tracking
use tracing::{info, instrument};

#[instrument(skip(sample_rate, data))]
pub fn memory_tracking(sample_rate: usize, data: &[f32]) {
    let bytes = data.len() * mem::size_of::<f32>();
    info!("Memory: {} MB", bytes / (1024 * 1024));
}
```

#### 5.4 Latency Distribution
```
P50: <100ms
P90: <150ms  
P95: <200ms
P99: <300ms

Python: P99 ~800ms (иногда генерация виснет)
Rust цель: P99 <300ms (детерминированно!)
```

## Quick Wins (2-3 days implementation)

### Win 1: Integrated Profiler (Day 1)

**файл:** `crates/runtime/src/profiler.rs`

```rust
// crates/runtime/src/profiler.rs
use std::time::Instant;
use tracing::{info, warn};

#[derive(Default)]
pub struct Profiler {
    sections: Vec<(&'static str, Duration)>,
}

impl Profiler {
    pub fn section<'a>(&'a mut self, name: &'static str) -> ProfilerGuard<'a> {
        ProfilerGuard { name, start: Instant::now(), profiler: self }
    }
    
    pub fn summary(&self) {
        let total: Duration = self.sections.iter().map(|(_, d)| *d).sum();
        info!("=== PROFILER SUMMARY ===");
        info!("Total: {:?}", total);
        for (name, duration) in &self.sections {
            let pct = *duration as f64 / total.as_secs_f64() * 100.0;
            info!("{:20} {:8.2?} ({:5.1}%)", name, duration, pct);
            if pct > 30.0 {
                warn!("  ⚠️  BOTTLENECK DETECTED!");
            }
        }
    }
}

pub struct ProfilerGuard<'a> {
    name: &'static str,
    start: Instant,
    profiler: &'a mut Profiler,
}

impl<'a> Drop for ProfilerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.sections.push((self.name, duration));
    }
}
```

**Интеграция в pipeline:**

```rust
// crates/runtime/src/pipeline.rs
use crate::profiler::Profiler;

pub struct TtsPipeline {
    profiler: Profiler,
    // ...
}

impl TtsPipeline {
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut profiler = Profiler::default();
        
        {
            let _p = profiler.section("normalization");
            self.normalize(text)?;
        }
        
        {
            let _p = profiler.section("tokenization");
            let tokens = self.tokenize(text)?;
        }
        
        {
            let _p = profiler.section("model_forward");
            let codes = self.model.forward(&tokens)?;
        }
        
        {
            let _p = profiler.section("codec_decode");
            let audio = self.codec.decode(&codes)?;
        }
        
        profiler.summary();
        Ok(audio)
    }
}
```

### Win 2: Reduce Allocations (Day 1-2)

**Файл:** `crates/audio-codec-12hz/src/allocator.rs`

```rust
use std::sync::Mutex;
use std::collections::VecDeque;

pub struct ReusableBuffer<T> {
    buffers: Mutex<VecDeque<Vec<T>>>,
    capacity: usize,
}

impl<T: Clone> ReusableBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffers: Mutex::new(VecDeque::with_capacity(4)),
            capacity,
        }
    }
    
    pub fn get(&self) -> Vec<T> {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(mut buf) = buffers.pop_back() {
            buf.clear();
            buf
        } else {
            Vec::with_capacity(self.capacity)
        }
    }
    
    pub fn return_buffer(&self, mut buffer: Vec<T>) {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.len() < 4 {
            buffer.clear();
            buffers.push_back(buffer);
        }
    }
}

// Использование
let mut buffer = reusable.get();
// ... работа с buffer ...
reusable.return_buffer(buffer);
```

**Результат:**
- Уменьшить GC pressure на 50%+
- Более детерминированный latency

### Win 3: Metal Shader Caching (Day 2)

**Файл:** `crates/runtime/src/metal_cache.rs`

```rust
use candle_core::{Device, Result as CandleResult};
use candle_metal::MetalDevice;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

pub struct MetalShaderRegistry {
    shaders: Mutex<HashMap<String, CachedShader>>,
    cache_dir: PathBuf,
}

struct CachedShader {
    library: Arc<CompiledMetalLibrary>,
    compiled_at: std::time::SystemTime,
}

impl MetalShaderRegistry {
    pub fn new() -> Self {
        let cache_dir = std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".cache/rusttts-metal"));
        
        std::fs::create_dir_all(&cache_dir).ok();
        
        Self {
            shaders: Mutex::new(HashMap::new()),
            cache_dir,
        }
    }
    
    pub fn get_or_compile(&self, source: &str, name: &str) -> Result<Arc<CompiledMetalLibrary>> {
        // Check cache
        {
            let shaders = self.shaders.lock().unwrap();
            if let Some(cached) = shaders.get(name) {
                // Check age (< 1 week = use cached)
                let age = cached.compiled_at.elapsed().unwrap().as_secs();
                if age < 604800 {  // 7 days
                    return Ok(cached.library.clone());
                }
            }
        }
        
        // Compile new shader
        let device = self.device();
        let lib = device.compile_shader(source)?;
        
        // Cache to disk
        let cache_file = self.cache_dir.join(format!("{}.bincache", name));
        bincode::serialize_into(
            &std::fs::File::create(cache_file)?,
            &lib.serialize()?
        )?;
        
        // Store in memory cache
        let cached = CachedShader {
            library: Arc::new(lib),
            compiled_at: std::time::SystemTime::now(),
        };
        
        let mut shaders = self.shaders.lock().unwrap();
        shaders.insert(name.into(), cached);
        
        Ok(cached.library)
    }
}
```

**Результат:**
- Убрать compile overhead на первом запуске
- Faster subsequent runs (cold start: 1.3s → 0.3s)

### Win 4: Pre-Warm Models (Day 2-3)

**Файл:** `crates/runtime/src/warm.rs`

```rust
pub use candle_core::Result;

pub async fn warm_model_cache(model: &Model, sample_text: &str) -> Result<()> {
    info!("Warming cache with sample synthesis...");
    
    // 5 warmup runs
    for i in 0..5 {
        info!("Warmup run {}/5", i + 1);
        model.synthesize(sample_text)?;
    }
    
    info!("Cold start pre-warming complete");
    Ok(())
}
```

**Интеграция при service startup:**

```rust
// crates/tts-server/src/main.rs
#[tokio::main]
async fn main() -> Result<()> {
    let model = Model::load(...)?;
    
    // Warm on startup
    warm_model_cache(&model, "Текст пример для прогрева.")?;
    
    // Start server
    start_server(model).await?;
}
```

**Результат:**
- Первый user request обрабатывается быстро
- Model JIT compilation происходит при startup, не при первом request

### Win 5: Tracing Integration (Day 3)

**Файл:** `crates/runtime/src/tracing_setup.rs`

```rust
use tracing_subscriber::{filter, fmt, EnvFilter};

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .compact()
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("rusttts=debug".parse().unwrap())
        )
        .init();
}
```

**В main.rs:**
```rust
#[tokio::main] 
async fn main() {
    runtime::tracing_setup::init_tracing();
    // ...
}
```

## Implementation Timeline

| Week | Tasks | Target |
|------|-------|--------|
| **Week 1** | Profiler, Allocations, Metal Cache, Tracing | Быстрые wins завершены |
| **Week 2** | Metal kernel optimization + SIMD | Metal RTF под 3.5x, CPU 3.5x |
| **Week 3-4** | INT8 Quantization | RTF под 2.5x (догнать/превзойти Python!) |
| **Week 5** | Fused kernels + Ring buffer cache | Ещё 20% ускорение |
| **Week 6** | Real metrics (TTFA, streaming) | Production-ready |

## Success Criteria

| Metric | Сейчас | Target | Status |
|--------|--------|--------|--------|
| Metal RTF (short) | 4.24x | < 3.0x | ❌ |
| CPU RTF (medium) | 2.81x | < 2.5x | ⚠️ |
| Time-to-first-audio | ??? | < 100ms | ❌ |
| Cold start | 1.3s | < 0.5s | ⚠️ |
| Memory (peak) | 1.5GB | < 700MB | ❌ |
| Binary size | 12MB | < 10MB | ⚠️ |

## Notes

### Замеры (tts-cli synth, real time)

| Модель | Устройство | Время (сек) | Примечание |
|--------|------------|-------------|------------|
| Q4 GGUF | CPU | 14.66 | Полный прогон (включая загрузку) |
| Q8 GGUF | CPU | 10.28 | Полный прогон (включая загрузку) |
| Q4 GGUF | Metal | 16.81 | Полный прогон (включая загрузку) |
| Q8 GGUF | Metal | 10.12 | Полный прогон (включая загрузку) |

- **Не сраться с CPU vs GPU**: если есть GPU - используй его!
- **Quantization даст наибольший прирост**: INT8 2.5x быстрее f32
- **Metal kernel optimization важен для Apple Silicon**: 30%+ ускорение
- **Architecture оптимisations (fused kernels)**: лёгкие wins 10-20%
- **Реальные метрики**: RTF ≠ latency streaming. TTFA + throughput важнее!

## Resources

- Candle quantization docs: https://huggingface.co/docs/candle/main/guides/quantization
- Metal Performance Shaders: https://apple.github.io/metal-shading-language/
- ARM NEON intrinsics: https://developer.arm.com/architecture/instruction-sets/intrinsics
- FFlash for Metal: https://github.com/microsoft/Flash-Attention