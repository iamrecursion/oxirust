//! Frame timing measurement.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Frame timing statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameStats {
    /// Average frame time.
    pub avg_frame_time: Duration,

    /// Minimum frame time.
    pub min_frame_time: Duration,

    /// Maximum frame time.
    pub max_frame_time: Duration,

    /// 1st percentile (p1).
    pub p1: Duration,

    /// 99th percentile (p99).
    pub p99: Duration,

    /// Average FPS.
    pub avg_fps: f64,

    /// Number of frames measured.
    pub frame_count: usize,

    /// Frame time variance.
    pub variance: f64,
}

/// Frame timer for measuring frame times.
#[derive(Debug)]
pub struct FrameTimer {
    frame_times: VecDeque<Duration>,
    max_samples: usize,
    current_frame_start: Option<Instant>,
    frame_count: u64,
}

impl FrameTimer {
    /// Create a new frame timer.
    pub fn new(max_samples: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(max_samples),
            max_samples,
            current_frame_start: None,
            frame_count: 0,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.current_frame_start = Some(Instant::now());
    }

    /// End the current frame.
    pub fn end_frame(&mut self) {
        if let Some(start) = self.current_frame_start.take() {
            let duration = start.elapsed();
            self.frame_times.push_back(duration);

            if self.frame_times.len() > self.max_samples {
                self.frame_times.pop_front();
            }

            self.frame_count += 1;
        }
    }

    /// Get frame statistics.
    pub fn stats(&self) -> FrameStats {
        if self.frame_times.is_empty() {
            return FrameStats {
                avg_frame_time: Duration::ZERO,
                min_frame_time: Duration::ZERO,
                max_frame_time: Duration::ZERO,
                p1: Duration::ZERO,
                p99: Duration::ZERO,
                avg_fps: 0.0,
                frame_count: 0,
                variance: 0.0,
            };
        }

        let mut sorted: Vec<_> = self.frame_times.iter().copied().collect();
        sorted.sort();

        let min_frame_time = sorted[0];
        let max_frame_time = sorted[sorted.len() - 1];

        let p1_idx = (sorted.len() as f64 * 0.01) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;

        let p1 = sorted.get(p1_idx).copied().unwrap_or(min_frame_time);
        let p99 = sorted.get(p99_idx).copied().unwrap_or(max_frame_time);

        let total: Duration = self.frame_times.iter().sum();
        let avg_frame_time = total / self.frame_times.len() as u32;

        let avg_fps = if avg_frame_time.as_secs_f64() > 0.0 {
            1.0 / avg_frame_time.as_secs_f64()
        } else {
            0.0
        };

        let mean = avg_frame_time.as_secs_f64();
        let variance = self
            .frame_times
            .iter()
            .map(|&t| {
                let diff = t.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.frame_times.len() as f64;

        FrameStats {
            avg_frame_time,
            min_frame_time,
            max_frame_time,
            p1,
            p99,
            avg_fps,
            frame_count: self.frame_times.len(),
            variance,
        }
    }

    /// Get the total number of frames measured.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the last N frame times.
    pub fn last_n_frames(&self, n: usize) -> Vec<Duration> {
        self.frame_times.iter().rev().take(n).copied().collect()
    }

    /// Clear all frame times.
    pub fn clear(&mut self) {
        self.frame_times.clear();
        self.current_frame_start = None;
    }

    /// Get maximum sample count.
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_timer() {
        let mut timer = FrameTimer::new(100);
        assert_eq!(timer.frame_count(), 0);

        timer.begin_frame();
        std::thread::sleep(Duration::from_millis(1));
        timer.end_frame();

        assert_eq!(timer.frame_count(), 1);
    }

    #[test]
    fn test_frame_stats() {
        let mut timer = FrameTimer::new(100);

        for _ in 0..10 {
            timer.begin_frame();
            std::thread::sleep(Duration::from_millis(1));
            timer.end_frame();
        }

        let stats = timer.stats();
        assert_eq!(stats.frame_count, 10);
        assert!(stats.avg_frame_time > Duration::ZERO);
        assert!(stats.avg_fps > 0.0);
    }

    #[test]
    fn test_frame_timer_max_samples() {
        let mut timer = FrameTimer::new(5);

        for _ in 0..10 {
            timer.begin_frame();
            timer.end_frame();
        }

        let stats = timer.stats();
        assert_eq!(stats.frame_count, 5); // Only keeps last 5
        assert_eq!(timer.frame_count(), 10); // But total count is 10
    }

    #[test]
    fn test_percentiles() {
        let mut timer = FrameTimer::new(100);

        for _ in 0..100 {
            timer.begin_frame();
            timer.end_frame();
        }

        let stats = timer.stats();
        assert!(stats.p1 <= stats.avg_frame_time);
        assert!(stats.p99 >= stats.avg_frame_time);
    }

    #[test]
    fn test_last_n_frames() {
        let mut timer = FrameTimer::new(100);

        for _ in 0..10 {
            timer.begin_frame();
            timer.end_frame();
        }

        let last_5 = timer.last_n_frames(5);
        assert_eq!(last_5.len(), 5);
    }
}
