//! Audio codec decode benchmarks.

use audio_codec_12hz::crossfade::Crossfader;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_crossfade(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossfade");

    // Test different fade durations
    for fade_ms in [5.0, 10.0, 20.0].iter() {
        let sample_rate = 24000;
        let chunk_size = (sample_rate as f32 * 0.05) as usize; // 50ms chunks
        let chunk: Vec<f32> = (0..chunk_size).map(|i| (i as f32 / 100.0).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("process", format!("{}ms", fade_ms)),
            fade_ms,
            |b, &fade_ms| {
                let mut crossfader = Crossfader::new(fade_ms, sample_rate);
                b.iter(|| crossfader.process(black_box(&chunk)))
            },
        );
    }

    group.finish();
}

fn bench_hann_window(c: &mut Criterion) {
    use audio_codec_12hz::crossfade::apply_hann_window;

    let mut group = c.benchmark_group("hann_window");

    for size in [240, 480, 2400].iter() {
        let samples: Vec<f32> = (0..*size).map(|i| (i as f32 / 100.0).sin()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut s = samples.clone();
                apply_hann_window(black_box(&mut s));
                s
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_crossfade, bench_hann_window);
criterion_main!(benches);
