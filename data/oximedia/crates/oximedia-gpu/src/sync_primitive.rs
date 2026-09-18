//! GPU synchronisation primitives (semaphores, fences, barriers).
#![allow(dead_code)]

/// Type of synchronisation primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncType {
    /// Binary semaphore: signals when GPU work completes.
    Semaphore,
    /// CPU-visible fence: CPU can wait on GPU progress.
    Fence,
    /// Pipeline barrier: enforces ordering within a command buffer.
    Barrier,
}

impl SyncType {
    /// Returns the category of wait operation this primitive uses.
    ///
    /// - `Semaphore` → `"gpu_wait"` (GPU waits on GPU)
    /// - `Fence`     → `"cpu_wait"` (CPU waits on GPU)
    /// - `Barrier`   → `"pipeline_stall"` (in-command serialisation)
    #[must_use]
    pub fn wait_type(&self) -> &'static str {
        match self {
            Self::Semaphore => "gpu_wait",
            Self::Fence => "cpu_wait",
            Self::Barrier => "pipeline_stall",
        }
    }

    /// Returns `true` for primitives the CPU can directly observe.
    #[must_use]
    pub fn is_cpu_visible(&self) -> bool {
        matches!(self, Self::Fence)
    }
}

/// State of a GPU sync object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    /// Initial / reset state – no work has been queued.
    Unsignaled,
    /// Work submitted to the GPU; may or may not be complete.
    Pending,
    /// GPU work has completed; primitive is signaled.
    Signaled,
}

impl SyncState {
    /// Returns `true` if the primitive is currently signaled.
    #[must_use]
    pub fn is_signaled(&self) -> bool {
        matches!(self, Self::Signaled)
    }

    /// Returns `true` if the primitive has in-flight GPU work.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }
}

/// A simulated GPU synchronisation object.
///
/// In a real driver backend this would wrap an underlying API object
/// (`VkSemaphore`, `VkFence`, `MTLEvent`, etc.).  Here we simulate
/// the state machine for testing and integration purposes.
pub struct GpuSync {
    sync_type: SyncType,
    state: SyncState,
    label: String,
    /// How many times this primitive has been signaled in its lifetime.
    signal_count: u64,
}

impl GpuSync {
    /// Create a new sync primitive in the `Unsignaled` state.
    #[must_use]
    pub fn new(sync_type: SyncType, label: impl Into<String>) -> Self {
        Self {
            sync_type,
            state: SyncState::Unsignaled,
            label: label.into(),
            signal_count: 0,
        }
    }

    /// Signal the primitive (simulates GPU work completing).
    ///
    /// Transitions `Unsignaled` → `Pending` → `Signaled`, or moves
    /// directly from `Pending` to `Signaled`.
    pub fn signal(&mut self) {
        self.state = SyncState::Signaled;
        self.signal_count += 1;
    }

    /// Mark the primitive as having pending GPU work queued.
    pub fn enqueue(&mut self) {
        if self.state == SyncState::Unsignaled {
            self.state = SyncState::Pending;
        }
    }

    /// Block (simulate) until the primitive is signaled.
    ///
    /// In this simulation, we simply check the current state.  Returns
    /// `true` if the primitive is (or was already) signaled, `false`
    /// if it is still `Unsignaled` (nothing was enqueued).
    #[must_use]
    pub fn wait(&self) -> bool {
        self.state == SyncState::Signaled
    }

    /// Reset the primitive back to `Unsignaled` so it can be re-used.
    ///
    /// Returns `false` if the primitive is still `Pending` (cannot safely
    /// reset GPU-side work that may be in flight).
    pub fn reset(&mut self) -> bool {
        if self.state == SyncState::Pending {
            return false;
        }
        self.state = SyncState::Unsignaled;
        true
    }

    /// Current state of the primitive.
    #[must_use]
    pub fn state(&self) -> SyncState {
        self.state
    }

    /// Type of this primitive.
    #[must_use]
    pub fn sync_type(&self) -> SyncType {
        self.sync_type
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Number of times this primitive has been signaled over its lifetime.
    #[must_use]
    pub fn signal_count(&self) -> u64 {
        self.signal_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SyncType tests ---

    #[test]
    fn test_semaphore_wait_type() {
        assert_eq!(SyncType::Semaphore.wait_type(), "gpu_wait");
    }

    #[test]
    fn test_fence_wait_type() {
        assert_eq!(SyncType::Fence.wait_type(), "cpu_wait");
    }

    #[test]
    fn test_barrier_wait_type() {
        assert_eq!(SyncType::Barrier.wait_type(), "pipeline_stall");
    }

    #[test]
    fn test_fence_is_cpu_visible() {
        assert!(SyncType::Fence.is_cpu_visible());
    }

    #[test]
    fn test_semaphore_not_cpu_visible() {
        assert!(!SyncType::Semaphore.is_cpu_visible());
    }

    // --- SyncState tests ---

    #[test]
    fn test_signaled_is_signaled() {
        assert!(SyncState::Signaled.is_signaled());
    }

    #[test]
    fn test_unsignaled_not_signaled() {
        assert!(!SyncState::Unsignaled.is_signaled());
    }

    #[test]
    fn test_pending_is_pending() {
        assert!(SyncState::Pending.is_pending());
    }

    #[test]
    fn test_signaled_not_pending() {
        assert!(!SyncState::Signaled.is_pending());
    }

    // --- GpuSync tests ---

    #[test]
    fn test_new_sync_is_unsignaled() {
        let s = GpuSync::new(SyncType::Fence, "f");
        assert_eq!(s.state(), SyncState::Unsignaled);
    }

    #[test]
    fn test_signal_transitions_to_signaled() {
        let mut s = GpuSync::new(SyncType::Semaphore, "s");
        s.signal();
        assert_eq!(s.state(), SyncState::Signaled);
    }

    #[test]
    fn test_wait_returns_true_when_signaled() {
        let mut s = GpuSync::new(SyncType::Fence, "f");
        s.signal();
        assert!(s.wait());
    }

    #[test]
    fn test_wait_returns_false_when_unsignaled() {
        let s = GpuSync::new(SyncType::Fence, "f");
        assert!(!s.wait());
    }

    #[test]
    fn test_reset_from_signaled_succeeds() {
        let mut s = GpuSync::new(SyncType::Fence, "f");
        s.signal();
        assert!(s.reset());
        assert_eq!(s.state(), SyncState::Unsignaled);
    }

    #[test]
    fn test_reset_from_pending_fails() {
        let mut s = GpuSync::new(SyncType::Semaphore, "s");
        s.enqueue();
        assert!(!s.reset());
        assert_eq!(s.state(), SyncState::Pending);
    }

    #[test]
    fn test_enqueue_transitions_to_pending() {
        let mut s = GpuSync::new(SyncType::Barrier, "b");
        s.enqueue();
        assert_eq!(s.state(), SyncState::Pending);
    }

    #[test]
    fn test_signal_count_increments() {
        let mut s = GpuSync::new(SyncType::Fence, "f");
        s.signal();
        s.reset();
        s.signal();
        assert_eq!(s.signal_count(), 2);
    }

    #[test]
    fn test_label_stored() {
        let s = GpuSync::new(SyncType::Fence, "my_fence");
        assert_eq!(s.label(), "my_fence");
    }

    #[test]
    fn test_sync_type_stored() {
        let s = GpuSync::new(SyncType::Barrier, "b");
        assert_eq!(s.sync_type(), SyncType::Barrier);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut s = GpuSync::new(SyncType::Fence, "lifecycle");
        assert_eq!(s.state(), SyncState::Unsignaled);
        s.enqueue();
        assert_eq!(s.state(), SyncState::Pending);
        s.signal();
        assert!(s.wait());
        assert!(s.reset());
        assert_eq!(s.state(), SyncState::Unsignaled);
    }
}
