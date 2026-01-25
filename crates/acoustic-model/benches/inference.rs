//! Acoustic model inference benchmarks.

use acoustic_model::sampling::{Sampler, SamplingConfig};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_sampling(c: &mut Criterion) {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 50,
        top_p: 0.9,
        seed: Some(42),
    };

    let logits: Vec<f32> = (0..65536).map(|i| (i as f32 / 1000.0).sin()).collect();

    let mut group = c.benchmark_group("sampling");

    group.bench_function("greedy", |b| {
        let sampler = Sampler::new(config.clone());
        b.iter(|| sampler.greedy(black_box(&logits)))
    });

    group.bench_function("top_k_p", |b| {
        let mut sampler = Sampler::new(config.clone());
        b.iter(|| sampler.sample(black_box(&logits)))
    });

    group.finish();
}

fn bench_softmax_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [1000, 10000, 65536, 151936].iter() {
        let logits: Vec<f32> = (0..*size).map(|i| (i as f32 / 1000.0).sin()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let config = SamplingConfig::default();
            let mut sampler = Sampler::new(config);
            b.iter(|| sampler.sample(black_box(&logits)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sampling, bench_softmax_sizes);
criterion_main!(benches);
