use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{Device, Result as CandleResult};
use once_cell::sync::Lazy;

pub struct MetalShaderCache {
    shaders: Mutex<HashMap<String, CachedShader>>,
    #[allow(dead_code)]
    cache_dir: PathBuf,
}

struct CachedShader {
    library: Arc<CompiledMetalLibrary>,
    compiled_at: std::time::SystemTime,
}

pub struct CompiledMetalLibrary;

impl MetalShaderCache {
    pub fn new() -> Self {
        let cache_dir = Self::default_cache_dir();
        std::fs::create_dir_all(&cache_dir).ok();

        Self {
            shaders: Mutex::new(HashMap::new()),
            cache_dir,
        }
    }

    fn default_cache_dir() -> PathBuf {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(xdg).join("rusttts-metal");
        }
        if cfg!(unix) {
            PathBuf::from(".cache/rusttts-metal")
        } else if cfg!(windows) {
            PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string()))
                .join("rusttts-metal")
        } else {
            PathBuf::from("cache/rusttts-metal")
        }
    }

    pub fn get_or_compile(
        &self,
        _source: &str,
        name: &str,
        _device: &Device,
    ) -> CandleResult<Arc<CompiledMetalLibrary>> {
        {
            let shaders = self.shaders.lock().unwrap();
            if let Some(cached) = shaders.get(name) {
                let age = cached.compiled_at.elapsed().unwrap().as_secs();
                if age < 604800 {
                    return Ok(cached.library.clone());
                }
            }
        }

        let library = Arc::new(CompiledMetalLibrary);
        let cached = CachedShader {
            library: library.clone(),
            compiled_at: std::time::SystemTime::now(),
        };

        let mut shaders = self.shaders.lock().unwrap();
        shaders.insert(name.into(), cached);

        Ok(library)
    }

    pub fn warm_cache_from_disk(&self, _device: &Device) -> CandleResult<()> {
        Ok(())
    }
}

pub static SHADER_CACHE: Lazy<Mutex<MetalShaderCache>> =
    Lazy::new(|| Mutex::new(MetalShaderCache::new()));

pub fn get_shader_cache() -> &'static Lazy<Mutex<MetalShaderCache>> {
    &SHADER_CACHE
}

pub fn warm_cache(device: &Device) -> CandleResult<()> {
    let cache = get_shader_cache();
    let cache = cache.lock().unwrap();
    cache.warm_cache_from_disk(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache = MetalShaderCache::new();
        assert!(cache.cache_dir.exists());
    }
}
