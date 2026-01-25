//! KV cache implementation for efficient autoregressive generation.

use parking_lot::RwLock;
use std::collections::HashMap;
use uuid::Uuid;

/// A handle to a cached KV state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheHandle {
    session_id: Uuid,
    layer: usize,
}

impl CacheHandle {
    /// Create a new cache handle.
    pub fn new(session_id: Uuid, layer: usize) -> Self {
        Self { session_id, layer }
    }
}

/// Entry in the KV cache.
#[derive(Debug)]
pub struct CacheEntry {
    /// Key tensor data (flattened).
    pub keys: Vec<f32>,
    /// Value tensor data (flattened).
    pub values: Vec<f32>,
    /// Current sequence length.
    pub seq_len: usize,
    /// Maximum sequence length.
    pub max_len: usize,
    /// Last access timestamp (for LRU eviction).
    pub last_access: std::time::Instant,
}

impl CacheEntry {
    /// Create a new cache entry with given capacity.
    pub fn new(max_len: usize, hidden_size: usize, num_kv_heads: usize) -> Self {
        let capacity = max_len * num_kv_heads * (hidden_size / num_kv_heads);
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            seq_len: 0,
            max_len,
            last_access: std::time::Instant::now(),
        }
    }

    /// Check if the cache is full.
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_len
    }

    /// Update the last access time.
    pub fn touch(&mut self) {
        self.last_access = std::time::Instant::now();
    }
}

/// KV cache manager with LRU eviction.
#[derive(Debug)]
pub struct KvCacheManager {
    entries: RwLock<HashMap<CacheHandle, CacheEntry>>,
    max_entries: usize,
    hidden_size: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
}

impl KvCacheManager {
    /// Create a new KV cache manager.
    pub fn new(
        max_entries: usize,
        hidden_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_entries)),
            max_entries,
            hidden_size,
            num_kv_heads,
            max_seq_len,
        }
    }

    /// Get or create a cache entry for the given handle.
    pub fn get_or_create(&self, handle: CacheHandle) -> CacheHandle {
        let mut entries = self.entries.write();

        if !entries.contains_key(&handle) {
            // Evict if necessary
            if entries.len() >= self.max_entries {
                self.evict_lru(&mut entries);
            }

            entries.insert(
                handle,
                CacheEntry::new(self.max_seq_len, self.hidden_size, self.num_kv_heads),
            );
        }

        handle
    }

    /// Remove a session's cache entries.
    pub fn remove_session(&self, session_id: Uuid) {
        let mut entries = self.entries.write();
        entries.retain(|k, _| k.session_id != session_id);
    }

    /// Get the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear all cache entries.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Evict the least recently used entry.
    fn evict_lru(&self, entries: &mut HashMap<CacheHandle, CacheEntry>) {
        if let Some((&handle, _)) = entries.iter().min_by_key(|(_, entry)| entry.last_access) {
            entries.remove(&handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_handle() {
        let session_id = Uuid::new_v4();
        let handle1 = CacheHandle::new(session_id, 0);
        let handle2 = CacheHandle::new(session_id, 0);
        let handle3 = CacheHandle::new(session_id, 1);

        assert_eq!(handle1, handle2);
        assert_ne!(handle1, handle3);
    }

    #[test]
    fn test_cache_manager_basic() {
        let manager = KvCacheManager::new(10, 256, 4, 1024);

        assert!(manager.is_empty());

        let session_id = Uuid::new_v4();
        let handle = CacheHandle::new(session_id, 0);
        manager.get_or_create(handle);

        assert_eq!(manager.len(), 1);
        assert!(!manager.is_empty());
    }

    #[test]
    fn test_cache_manager_session_removal() {
        let manager = KvCacheManager::new(10, 256, 4, 1024);

        let session_id = Uuid::new_v4();
        for layer in 0..5 {
            let handle = CacheHandle::new(session_id, layer);
            manager.get_or_create(handle);
        }

        assert_eq!(manager.len(), 5);

        manager.remove_session(session_id);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_cache_entry_is_full() {
        let mut entry = CacheEntry::new(100, 256, 4);
        assert!(!entry.is_full());

        entry.seq_len = 100;
        assert!(entry.is_full());
    }
}
