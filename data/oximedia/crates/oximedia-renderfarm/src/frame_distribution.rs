//! Frame distribution across render farm nodes.

#![allow(clippy::cast_precision_loss)]

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use thiserror::Error;

/// Strategy used to distribute frames across nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionStrategy {
    /// Assign frames in simple round-robin order.
    RoundRobin,
    /// Assign frames based on current node load.
    LoadBased,
    /// Randomly spread frames across nodes.
    RandomSpread,
    /// Use scene complexity hints to group frames.
    SceneAware,
}

impl DistributionStrategy {
    /// Returns `true` if this strategy analyses scene content for decisions.
    #[must_use]
    pub fn considers_content(&self) -> bool {
        matches!(self, Self::SceneAware)
    }
}

/// An inclusive range of frame numbers `[start, end]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRange {
    /// First frame (inclusive).
    pub start: u32,
    /// Last frame (inclusive).
    pub end: u32,
}

impl FrameRange {
    /// Creates a new `FrameRange`.  Panics in debug builds if `end < start`.
    #[must_use]
    pub fn new(start: u32, end: u32) -> Self {
        debug_assert!(end >= start, "end must be >= start");
        Self { start, end }
    }

    /// Returns the number of frames in this range (inclusive).
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.end.saturating_sub(self.start) + 1
    }

    /// Returns `true` if frame `f` falls within `[start, end]`.
    #[must_use]
    pub fn contains(&self, f: u32) -> bool {
        f >= self.start && f <= self.end
    }

    /// Splits this range into `n` roughly equal sub-ranges.
    ///
    /// If `n == 0` or the range is empty, an empty `Vec` is returned.
    /// Remainder frames are distributed to the first sub-ranges.
    #[must_use]
    pub fn split(&self, n: u32) -> Vec<FrameRange> {
        let total = self.frame_count();
        if n == 0 || total == 0 {
            return Vec::new();
        }
        let n = n.min(total); // can't have more parts than frames
        let base = total / n;
        let remainder = total % n;
        let mut ranges = Vec::with_capacity(n as usize);
        let mut cursor = self.start;
        for i in 0..n {
            let extra = u32::from(i < remainder);
            let count = base + extra;
            let end = cursor + count - 1;
            ranges.push(FrameRange::new(cursor, end));
            cursor = end + 1;
        }
        ranges
    }
}

/// The assignment of a frame range to a specific farm node.
#[derive(Debug, Clone)]
pub struct NodeAssignment {
    /// ID of the farm node.
    pub node_id: u32,
    /// The frames assigned to this node.
    pub frame_range: FrameRange,
    /// Estimated wall-clock time to complete the assignment (seconds).
    pub estimated_time_s: f32,
}

impl NodeAssignment {
    /// Creates a new `NodeAssignment`.
    #[must_use]
    pub fn new(node_id: u32, frame_range: FrameRange, estimated_time_s: f32) -> Self {
        Self {
            node_id,
            frame_range,
            estimated_time_s,
        }
    }

    /// Average frames per second for this assignment.
    ///
    /// Returns 0.0 if `estimated_time_s` is zero.
    #[must_use]
    pub fn frames_per_second(&self) -> f32 {
        if self.estimated_time_s <= 0.0 {
            return 0.0;
        }
        self.frame_range.frame_count() as f32 / self.estimated_time_s
    }
}

/// Distributes frames across farm nodes using a chosen strategy.
#[derive(Debug)]
pub struct FrameDistributor {
    /// The distribution strategy to use.
    pub strategy: DistributionStrategy,
}

impl FrameDistributor {
    /// Creates a new `FrameDistributor` with the given strategy.
    #[must_use]
    pub fn new(strategy: DistributionStrategy) -> Self {
        Self { strategy }
    }

    /// Distributes `total_frames` (starting from frame 1) across `node_count` nodes.
    ///
    /// Returns one `FrameRange` per node.  Empty if `node_count == 0`.
    #[must_use]
    pub fn distribute(&self, total_frames: u32, node_count: u32) -> Vec<FrameRange> {
        if node_count == 0 || total_frames == 0 {
            return Vec::new();
        }
        let full = FrameRange::new(1, total_frames);
        full.split(node_count)
    }

    /// Estimates total completion time in seconds.
    ///
    /// `avg_fps` is the per-node rendering speed.  Returns 0.0 for invalid inputs.
    #[must_use]
    pub fn estimate_completion_s(&self, total_frames: u32, avg_fps: f32, node_count: u32) -> f32 {
        if node_count == 0 || avg_fps <= 0.0 {
            return 0.0;
        }
        total_frames as f32 / (avg_fps * node_count as f32)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_not_content_aware() {
        assert!(!DistributionStrategy::RoundRobin.considers_content());
    }

    #[test]
    fn test_load_based_not_content_aware() {
        assert!(!DistributionStrategy::LoadBased.considers_content());
    }

    #[test]
    fn test_random_spread_not_content_aware() {
        assert!(!DistributionStrategy::RandomSpread.considers_content());
    }

    #[test]
    fn test_scene_aware_considers_content() {
        assert!(DistributionStrategy::SceneAware.considers_content());
    }

    #[test]
    fn test_frame_range_count() {
        let r = FrameRange::new(1, 100);
        assert_eq!(r.frame_count(), 100);
    }

    #[test]
    fn test_frame_range_single_frame() {
        let r = FrameRange::new(5, 5);
        assert_eq!(r.frame_count(), 1);
    }

    #[test]
    fn test_frame_range_contains_true() {
        let r = FrameRange::new(10, 20);
        assert!(r.contains(15));
    }

    #[test]
    fn test_frame_range_contains_boundary() {
        let r = FrameRange::new(10, 20);
        assert!(r.contains(10));
        assert!(r.contains(20));
    }

    #[test]
    fn test_frame_range_not_contains() {
        let r = FrameRange::new(10, 20);
        assert!(!r.contains(9));
        assert!(!r.contains(21));
    }

    #[test]
    fn test_frame_range_split_even() {
        let r = FrameRange::new(1, 10);
        let parts = r.split(5);
        assert_eq!(parts.len(), 5);
        for p in &parts {
            assert_eq!(p.frame_count(), 2);
        }
    }

    #[test]
    fn test_frame_range_split_covers_all_frames() {
        let r = FrameRange::new(1, 100);
        let parts = r.split(7);
        let total: u32 = parts.iter().map(super::FrameRange::frame_count).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_frame_range_split_zero_parts() {
        let r = FrameRange::new(1, 10);
        assert!(r.split(0).is_empty());
    }

    #[test]
    fn test_node_assignment_fps() {
        let a = NodeAssignment::new(0, FrameRange::new(1, 100), 50.0);
        assert!((a.frames_per_second() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_node_assignment_fps_zero_time() {
        let a = NodeAssignment::new(0, FrameRange::new(1, 10), 0.0);
        assert!((a.frames_per_second() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distributor_distribute_count() {
        let d = FrameDistributor::new(DistributionStrategy::RoundRobin);
        let parts = d.distribute(120, 4);
        assert_eq!(parts.len(), 4);
    }

    #[test]
    fn test_distributor_distribute_zero_nodes() {
        let d = FrameDistributor::new(DistributionStrategy::LoadBased);
        assert!(d.distribute(100, 0).is_empty());
    }

    #[test]
    fn test_distributor_estimate_completion() {
        let d = FrameDistributor::new(DistributionStrategy::SceneAware);
        // 100 frames, 2 fps per node, 5 nodes → 100 / (2 * 5) = 10 s
        let secs = d.estimate_completion_s(100, 2.0, 5);
        assert!((secs - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_distributor_estimate_zero_fps() {
        let d = FrameDistributor::new(DistributionStrategy::RoundRobin);
        assert!((d.estimate_completion_s(100, 0.0, 4) - 0.0).abs() < f32::EPSILON);
    }

    // --- BackpressureDistributor ---

    #[test]
    fn test_backpressure_distributor_basic_dispatch() {
        let dist = BackpressureDistributor::new(2, 4);
        dist.dispatch(FrameTask::new(1, 5, "job-a"))
            .expect("dispatch ok");
        // At least one of the two receivers should have the task.
        let task = dist
            .worker_receiver(0)
            .expect("r0 exists")
            .try_recv()
            .or_else(|_| dist.worker_receiver(1).expect("r1 exists").try_recv())
            .expect("task should be received");
        assert_eq!(task.frame, 1);
    }

    #[test]
    fn test_backpressure_distributor_no_workers() {
        let dist = BackpressureDistributor::new(0, 4);
        assert!(matches!(
            dist.dispatch(FrameTask::new(1, 1, "j")),
            Err(DispatchError::NoWorkers)
        ));
    }

    #[test]
    fn test_backpressure_distributor_try_dispatch_full() {
        // capacity = max(1 * 1, 1) = 1
        let dist = BackpressureDistributor::new(1, 1);
        dist.try_dispatch(FrameTask::new(1, 0, "j"))
            .expect("first ok");
        // Channel is now full; second should fail immediately.
        let result = dist.try_dispatch(FrameTask::new(2, 0, "j"));
        assert!(matches!(result, Err(DispatchError::AllFull)));
    }

    #[test]
    fn test_backpressure_distributor_capacity() {
        let dist = BackpressureDistributor::new(4, 3);
        assert_eq!(dist.capacity(), 12);
        assert_eq!(dist.worker_count(), 4);
    }

    #[test]
    fn test_backpressure_distributor_receiver_out_of_range() {
        let dist = BackpressureDistributor::new(2, 2);
        assert!(dist.worker_receiver(10).is_none());
    }
}

// ---------------------------------------------------------------------------
// BackpressureDistributor
// ---------------------------------------------------------------------------

/// A task representing a single frame to be rendered by a worker.
#[derive(Debug, Clone)]
pub struct FrameTask {
    /// The frame number to render.
    pub frame: u32,
    /// Priority hint — higher values are rendered first.
    pub priority: u8,
    /// Optional job identifier for correlation.
    pub job_id: String,
}

impl FrameTask {
    /// Create a new frame task.
    pub fn new(frame: u32, priority: u8, job_id: impl Into<String>) -> Self {
        Self {
            frame,
            priority,
            job_id: job_id.into(),
        }
    }
}

/// Errors that can occur when dispatching a frame task.
#[derive(Debug, Error)]
pub enum DispatchError {
    /// All worker channels are full (backpressure threshold reached).
    #[error("all worker channels are full (backpressure)")]
    AllFull,
    /// No workers have been registered.
    #[error("no workers registered")]
    NoWorkers,
    /// The requested worker index is out of range.
    #[error("worker index {0} out of range")]
    InvalidWorker(usize),
}

/// Distributes `FrameTask`s to workers using bounded crossbeam channels.
///
/// When all worker channels are full, [`BackpressureDistributor::dispatch`]
/// **blocks** until a slot opens, providing natural backpressure. The
/// non-blocking [`BackpressureDistributor::try_dispatch`] returns
/// [`DispatchError::AllFull`] immediately instead.
///
/// Worker selection: the least-loaded worker (fewest queued items) is chosen
/// on each dispatch call.
pub struct BackpressureDistributor {
    /// Per-worker channel capacity.
    capacity: usize,
    /// Per-worker senders.
    senders: Vec<Sender<FrameTask>>,
    /// Per-worker receivers handed out to workers via `worker_receiver`.
    receivers: Vec<Receiver<FrameTask>>,
}

impl BackpressureDistributor {
    /// Create a distributor for `workers` workers.
    ///
    /// Each worker's channel capacity = `workers × in_flight_factor` (minimum 1).
    pub fn new(workers: usize, in_flight_factor: usize) -> Self {
        let capacity = (workers * in_flight_factor).max(1);
        let mut senders = Vec::with_capacity(workers);
        let mut receivers = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (s, r) = bounded(capacity);
            senders.push(s);
            receivers.push(r);
        }
        Self {
            capacity,
            senders,
            receivers,
        }
    }

    /// Dispatch `task` to the least-loaded worker, **blocking** if its
    /// channel is full.
    ///
    /// Returns `Err(DispatchError::NoWorkers)` if no workers are registered.
    pub fn dispatch(&self, task: FrameTask) -> Result<(), DispatchError> {
        if self.senders.is_empty() {
            return Err(DispatchError::NoWorkers);
        }
        let idx = self.least_loaded_index();
        self.senders[idx]
            .send(task)
            .map_err(|_| DispatchError::AllFull)
    }

    /// Non-blocking dispatch: returns `Err(DispatchError::AllFull)` immediately
    /// if the least-loaded worker's channel is full.
    pub fn try_dispatch(&self, task: FrameTask) -> Result<(), DispatchError> {
        if self.senders.is_empty() {
            return Err(DispatchError::NoWorkers);
        }
        let idx = self.least_loaded_index();
        match self.senders[idx].try_send(task) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(DispatchError::AllFull),
            Err(TrySendError::Disconnected(_)) => Err(DispatchError::AllFull),
        }
    }

    /// Return a reference to the receiver for worker `worker_id`.
    ///
    /// Returns `None` if `worker_id` is out of range.
    pub fn worker_receiver(&self, worker_id: usize) -> Option<&Receiver<FrameTask>> {
        self.receivers.get(worker_id)
    }

    /// Channel capacity (same for all workers).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of workers.
    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }

    /// Index of the worker with the fewest queued tasks.
    fn least_loaded_index(&self) -> usize {
        self.senders
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.len())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
