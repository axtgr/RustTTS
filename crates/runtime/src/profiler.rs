use std::time::{Duration, Instant};
use tracing::{info, warn};

#[derive(Default)]
pub struct Profiler {
    sections: Vec<(&'static str, Duration)>,
}

impl Profiler {
    pub fn section<'a>(&'a mut self, name: &'static str) -> ProfilerGuard<'a> {
        info!(section = name, "Profiler section started");
        ProfilerGuard {
            name,
            start: Instant::now(),
            profiler: self,
        }
    }

    pub fn summary(&self) {
        let total: Duration = self.sections.iter().map(|(_, d)| *d).sum();
        info!("=== PROFILER SUMMARY ===");
        info!("Total: {:?}", total);
        if total == Duration::ZERO {
            info!("No profiler sections collected");
            return;
        }
        for (name, duration) in &self.sections {
            let pct = duration.as_secs_f64() / total.as_secs_f64() * 100.0;
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
        info!(section = self.name, elapsed = ?duration, "Profiler section finished");
        self.profiler.sections.push((self.name, duration));
    }
}
