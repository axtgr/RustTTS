use acoustic_model::{AcousticModelConfig, Attention, LayerKvCache, RotaryEmbedding};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_chunked_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    // Используем конфигурацию, близкую к реальной, но уменьшенную для скорости
    let mut config = AcousticModelConfig::tiny();
    config.hidden_size = 256;
    config.num_attention_heads = 4;
    config.num_kv_heads = 2; // GQA 2x
    config.head_dim = 64;

    let vb = VarBuilder::zeros(DType::F32, &device);
    let attn = Attention::new(&config, vb.pp("attn")).unwrap();
    let rope = RotaryEmbedding::new(config.head_dim, 4096, 10000.0, &device).unwrap();

    let chunk_size = 64;

    // Сценарий 1: Короткий контекст (128 токенов)
    {
        let mut group = c.benchmark_group("attention_step_short");
        let context_len = 128;
        let mut cache = LayerKvCache::new(
            chunk_size,
            config.head_dim,
            config.num_kv_heads,
            &device,
            DType::F32,
        )
        .unwrap();

        let k = Tensor::randn(
            0f32,
            1f32,
            (1, config.num_kv_heads, 1, config.head_dim),
            &device,
        )
        .unwrap();
        let v = Tensor::randn(
            0f32,
            1f32,
            (1, config.num_kv_heads, 1, config.head_dim),
            &device,
        )
        .unwrap();

        for _ in 0..context_len {
            cache.append(&k, &v).unwrap();
        }

        let x = Tensor::randn(0f32, 1f32, (1, 1, config.hidden_size), &device).unwrap();

        group.bench_function("chunked_attn_128", |b| {
            b.iter(|| {
                // Клонируем структуру кэша (Arc на тензоры), чтобы не менять состояние между итерациями
                let mut local_cache = cache.clone();
                let _ = attn
                    .forward(&x, &rope, context_len, Some(&mut local_cache))
                    .unwrap();
            });
        });
        group.finish();
    }

    // Сценарий 2: Длинный контекст (1024 токена)
    {
        let mut group = c.benchmark_group("attention_step_long");
        let context_len = 1024;
        let mut cache = LayerKvCache::new(
            chunk_size,
            config.head_dim,
            config.num_kv_heads,
            &device,
            DType::F32,
        )
        .unwrap();

        let k = Tensor::randn(
            0f32,
            1f32,
            (1, config.num_kv_heads, 1, config.head_dim),
            &device,
        )
        .unwrap();
        let v = Tensor::randn(
            0f32,
            1f32,
            (1, config.num_kv_heads, 1, config.head_dim),
            &device,
        )
        .unwrap();

        for _ in 0..context_len {
            cache.append(&k, &v).unwrap();
        }

        let x = Tensor::randn(0f32, 1f32, (1, 1, config.hidden_size), &device).unwrap();

        group.bench_function("chunked_attn_1024", |b| {
            b.iter(|| {
                let mut local_cache = cache.clone();
                let _ = attn
                    .forward(&x, &rope, context_len, Some(&mut local_cache))
                    .unwrap();
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_chunked_attention);
criterion_main!(benches);
