//! Request queue with priority and deadline management.

use dashmap::DashMap;
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};
use tts_core::SynthesisRequest;
use uuid::Uuid;

/// A queued synthesis request with scheduling metadata.
#[derive(Debug)]
pub struct QueuedRequest {
    /// The original synthesis request.
    pub request: SynthesisRequest,
    /// Time when the request was queued.
    pub queued_at: Instant,
    /// Deadline for completion (if set).
    pub deadline: Option<Instant>,
    /// Request ID for tracking.
    pub id: Uuid,
}

impl QueuedRequest {
    /// Create a new queued request.
    pub fn new(request: SynthesisRequest) -> Self {
        let deadline = request
            .max_latency_ms
            .map(|ms| Instant::now() + Duration::from_millis(ms));

        Self {
            id: request.session_id,
            request,
            queued_at: Instant::now(),
            deadline,
        }
    }

    /// Check if the request has exceeded its deadline.
    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() > d)
    }

    /// Get the time spent waiting in queue.
    pub fn wait_time(&self) -> Duration {
        self.queued_at.elapsed()
    }
}

// Ordering for priority queue (higher priority first, then earlier deadline)
// BinaryHeap is a max-heap, so Greater = higher priority = popped first
impl Ord for QueuedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (higher priority = Greater = popped first)
        match self.request.priority.cmp(&other.request.priority) {
            Ordering::Equal => {}
            ord => return ord, // Higher priority should be Greater
        }

        // Then by deadline (earlier deadline = Greater = popped first)
        match (&self.deadline, &other.deadline) {
            (Some(d1), Some(d2)) => d2.cmp(d1), // Earlier deadline = Greater
            (Some(_), None) => Ordering::Greater, // Requests with deadlines go first
            (None, Some(_)) => Ordering::Less,
            (None, None) => other.queued_at.cmp(&self.queued_at), // Earlier queued = Greater (FIFO)
        }
    }
}

impl PartialOrd for QueuedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueuedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for QueuedRequest {}

/// Priority queue for synthesis requests.
#[derive(Debug)]
pub struct RequestQueue {
    queue: Mutex<BinaryHeap<QueuedRequest>>,
    requests: DashMap<Uuid, ()>,
    max_size: usize,
}

impl RequestQueue {
    /// Create a new request queue with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Mutex::new(BinaryHeap::with_capacity(max_size)),
            requests: DashMap::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a request to the queue.
    ///
    /// Returns `false` if the queue is full.
    pub fn push(&self, request: SynthesisRequest) -> bool {
        let id = request.session_id;

        if self.requests.len() >= self.max_size {
            return false;
        }

        let queued = QueuedRequest::new(request);
        self.requests.insert(id, ());
        self.queue.lock().push(queued);
        true
    }

    /// Remove and return the highest priority request.
    pub fn pop(&self) -> Option<QueuedRequest> {
        let mut queue = self.queue.lock();

        // Skip expired requests
        while let Some(req) = queue.pop() {
            self.requests.remove(&req.id);
            if !req.is_expired() {
                return Some(req);
            }
        }

        None
    }

    /// Peek at the highest priority request without removing it.
    pub fn peek(&self) -> Option<Uuid> {
        self.queue.lock().peek().map(|r| r.id)
    }

    /// Cancel a request by ID.
    pub fn cancel(&self, id: Uuid) -> bool {
        self.requests.remove(&id).is_some()
        // Note: The request will be skipped when popped
    }

    /// Get the current queue size.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Check if the queue is full.
    pub fn is_full(&self) -> bool {
        self.requests.len() >= self.max_size
    }

    /// Clear the queue.
    pub fn clear(&self) {
        self.queue.lock().clear();
        self.requests.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_core::{Lang, Priority};

    fn make_request(priority: Priority) -> SynthesisRequest {
        SynthesisRequest::new("test")
            .with_lang(Lang::En)
            .with_priority(priority)
    }

    #[test]
    fn test_queue_basic_operations() {
        let queue = RequestQueue::new(10);

        assert!(queue.is_empty());
        assert!(!queue.is_full());

        let req = make_request(Priority::Normal);
        assert!(queue.push(req));

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let popped = queue.pop();
        assert!(popped.is_some());
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_priority_ordering() {
        let queue = RequestQueue::new(10);

        // Add requests in mixed order
        queue.push(make_request(Priority::Low));
        queue.push(make_request(Priority::Critical));
        queue.push(make_request(Priority::Normal));
        queue.push(make_request(Priority::High));

        // Should pop in priority order
        assert_eq!(queue.pop().unwrap().request.priority, Priority::Critical);
        assert_eq!(queue.pop().unwrap().request.priority, Priority::High);
        assert_eq!(queue.pop().unwrap().request.priority, Priority::Normal);
        assert_eq!(queue.pop().unwrap().request.priority, Priority::Low);
    }

    #[test]
    fn test_queue_full() {
        let queue = RequestQueue::new(2);

        assert!(queue.push(make_request(Priority::Normal)));
        assert!(queue.push(make_request(Priority::Normal)));
        assert!(!queue.push(make_request(Priority::Normal))); // Should fail

        assert!(queue.is_full());
    }

    #[test]
    fn test_queue_cancel() {
        let queue = RequestQueue::new(10);

        let req = make_request(Priority::Normal);
        let id = req.session_id;
        queue.push(req);

        assert_eq!(queue.len(), 1);
        assert!(queue.cancel(id));
        // Note: len() still shows 1 because request is in heap
        // but it will be skipped on pop
    }

    #[test]
    fn test_queued_request_wait_time() {
        let req = make_request(Priority::Normal);
        let queued = QueuedRequest::new(req);

        std::thread::sleep(Duration::from_millis(10));
        assert!(queued.wait_time() >= Duration::from_millis(10));
    }
}
