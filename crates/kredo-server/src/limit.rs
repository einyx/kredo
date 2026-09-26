//! Token-bucket rate limiter, keyed per client (API key or remote IP).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct RateLimiter {
    /// Requests per minute; 0 = disabled.
    per_min: f64,
    capacity: f64,
    buckets: Mutex<HashMap<String, Bucket>>,
}

struct Bucket {
    tokens: f64,
    updated: Instant,
}

impl RateLimiter {
    pub fn new(per_min: u32) -> Self {
        let per_min = f64::from(per_min);
        Self {
            per_min,
            capacity: per_min.max(1.0),
            buckets: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.per_min > 0.0
    }

    /// Allow one request for `key`? Refills lazily.
    pub fn allow(&self, key: &str) -> bool {
        if !self.enabled() {
            return true;
        }
        let mut buckets = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        let refill_per_sec = self.per_min / 60.0;
        let bucket = buckets.entry(key.to_string()).or_insert(Bucket {
            tokens: self.capacity,
            updated: now,
        });
        let elapsed = now.duration_since(bucket.updated).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * refill_per_sec).min(self.capacity);
        bucket.updated = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Drop idle buckets so long-running daemons don't grow the map forever.
    pub fn sweep(&self) -> usize {
        let mut buckets = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        let before = buckets.len();
        buckets.retain(|_, b| b.updated.elapsed() < Duration::from_secs(600));
        before - buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_up_to_capacity_then_blocks() {
        let rl = RateLimiter::new(5);
        for _ in 0..5 {
            assert!(rl.allow("client-a"));
        }
        assert!(!rl.allow("client-a"));
        // Independent buckets.
        assert!(rl.allow("client-b"));
    }

    #[test]
    fn disabled_never_blocks() {
        let rl = RateLimiter::new(0);
        for _ in 0..10_000 {
            assert!(rl.allow("x"));
        }
    }
}
