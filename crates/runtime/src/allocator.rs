use std::collections::VecDeque;
use std::sync::Mutex;

pub struct ReusableBufferPool<T> {
    buffers: Mutex<VecDeque<Vec<T>>>,
    capacity: usize,
}

impl<T> ReusableBufferPool<T> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_reuse() {
        let pool = ReusableBufferPool::with_capacity(1024);

        let buf1 = pool.get();
        drop(buf1);

        let buf2 = pool.get();
        drop(buf2);

        assert!(pool.buffers.lock().unwrap().len() <= 4);
    }
}
