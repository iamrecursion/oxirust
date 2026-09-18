//! Asynchronous GPU buffer transfer pipeline.
//!
//! Enables overlapping CPU computation with GPU data transfers by providing
//! a pipeline abstraction that accepts transfer requests, tracks their
//! completion through atomic flags, and simulates immediate completion in
//! CPU-fallback mode.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Transfer direction for async GPU operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Copy data from host (CPU) memory to device (GPU) memory.
    HostToDevice,
    /// Copy data from device (GPU) memory to host (CPU) memory.
    DeviceToHost,
    /// Copy data between two device (GPU) memory regions.
    DeviceToDevice,
}

/// A pending async transfer operation stored internally in the pipeline.
struct AsyncTransfer<T> {
    #[allow(dead_code)]
    data: Vec<T>,
    #[allow(dead_code)]
    direction: TransferDirection,
    handle: TransferHandle,
}

/// Error type for async transfer pipeline operations.
#[derive(Debug, Error)]
pub enum AsyncTransferError {
    /// Returned when a submit is attempted and the pipeline already holds
    /// `max_pending` unfinished transfers.
    #[error("Pipeline full: {0} pending transfers")]
    PipelineFull(usize),

    /// Returned when a transfer ID is queried but no matching handle exists.
    #[error("Transfer ID {0} not found")]
    NotFound(u64),

    /// Returned when the internal lock cannot be acquired.
    #[error("Failed to acquire pipeline lock")]
    LockError,
}

/// An opaque handle returned by [`AsyncTransferPipeline::submit`].
///
/// Use [`TransferHandle::is_complete`] to poll completion without blocking,
/// or pass the handle to [`AsyncTransferPipeline::flush`] to drain all pending
/// work.
#[derive(Debug, Clone)]
pub struct TransferHandle {
    /// Unique monotonically-increasing identifier for this transfer.
    pub id: u64,
    completed: Arc<AtomicBool>,
}

impl TransferHandle {
    /// Returns `true` when the underlying transfer has finished.
    ///
    /// In CPU simulation mode this is always `true` immediately after
    /// submission.
    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }
}

/// Pipeline that manages a bounded queue of in-flight async transfers.
///
/// In CPU simulation mode every submitted transfer completes synchronously
/// (the completion flag is set to `true` before `submit` returns).  This
/// means the pipeline compiles and passes tests on machines without real GPU
/// hardware.
///
/// # Type parameters
///
/// * `T` – The element type of the transfer buffers.  Must be `Clone + Send +
///   'static`.
pub struct AsyncTransferPipeline<T> {
    pending: Mutex<VecDeque<AsyncTransfer<T>>>,
    max_pending: usize,
    id_counter: AtomicU64,
}

impl<T: Clone + Send + 'static> AsyncTransferPipeline<T> {
    /// Create a new pipeline that allows up to `max_pending` in-flight
    /// transfers before returning [`AsyncTransferError::PipelineFull`].
    ///
    /// A `max_pending` of 0 is valid and will cause every `submit` to fail
    /// immediately.
    pub fn new(max_pending: usize) -> Self {
        Self {
            pending: Mutex::new(VecDeque::new()),
            max_pending,
            id_counter: AtomicU64::new(1),
        }
    }

    /// Submit a transfer request for `data` in the given `direction`.
    ///
    /// In CPU simulation mode the transfer completes immediately: the returned
    /// [`TransferHandle`] will already report `is_complete() == true`.
    ///
    /// # Errors
    ///
    /// Returns [`AsyncTransferError::PipelineFull`] when the number of
    /// currently pending (incomplete) transfers equals `max_pending`.
    pub fn submit(
        &self,
        data: Vec<T>,
        direction: TransferDirection,
    ) -> Result<TransferHandle, AsyncTransferError> {
        let mut queue = self
            .pending
            .lock()
            .map_err(|_| AsyncTransferError::LockError)?;

        // Count only incomplete transfers against the cap.
        let in_flight = queue.iter().filter(|t| !t.handle.is_complete()).count();
        if in_flight >= self.max_pending {
            return Err(AsyncTransferError::PipelineFull(in_flight));
        }

        let id = self.id_counter.fetch_add(1, Ordering::Relaxed);
        let completed = Arc::new(AtomicBool::new(false));
        let handle = TransferHandle {
            id,
            completed: Arc::clone(&completed),
        };

        // CPU simulation: mark the transfer as immediately complete.
        completed.store(true, Ordering::Release);

        queue.push_back(AsyncTransfer {
            data,
            direction,
            handle: handle.clone(),
        });

        Ok(handle)
    }

    /// Returns `true` if the transfer identified by `handle` has completed.
    ///
    /// This is a convenience wrapper around [`TransferHandle::is_complete`].
    pub fn is_complete(&self, handle: &TransferHandle) -> bool {
        handle.is_complete()
    }

    /// Block until all pending transfers have completed, then drain the queue.
    ///
    /// In CPU simulation mode this returns immediately because all transfers
    /// are marked complete on submission.
    ///
    /// # Errors
    ///
    /// Returns [`AsyncTransferError::LockError`] if the internal mutex cannot
    /// be acquired.
    pub fn flush(&self) -> Result<(), AsyncTransferError> {
        let mut queue = self
            .pending
            .lock()
            .map_err(|_| AsyncTransferError::LockError)?;

        // In CPU mode every transfer is already done; just drain the queue.
        // In a real GPU implementation this would call a device-synchronise API
        // and wait for each in-flight DMA operation.
        queue.retain(|transfer| !transfer.handle.is_complete());

        Ok(())
    }

    /// Return the number of transfers currently tracked in the pipeline
    /// (including already-completed ones that have not yet been flushed).
    pub fn pending_count(&self) -> usize {
        self.pending.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Return the number of transfers that have NOT yet completed.
    pub fn in_flight_count(&self) -> usize {
        self.pending
            .lock()
            .map(|q| q.iter().filter(|t| !t.handle.is_complete()).count())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Submitting a transfer should produce a handle that is immediately
    /// complete in CPU simulation mode.
    #[test]
    fn test_async_transfer_submit() {
        let pipeline: AsyncTransferPipeline<f32> = AsyncTransferPipeline::new(8);

        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let handle = pipeline
            .submit(data.clone(), TransferDirection::HostToDevice)
            .expect("submit should succeed");

        assert!(
            handle.is_complete(),
            "handle should be complete immediately in CPU mode"
        );
        assert!(
            pipeline.is_complete(&handle),
            "pipeline.is_complete should match handle"
        );
    }

    /// Submitting multiple transfers then flushing should leave an empty
    /// in-flight queue.
    #[test]
    fn test_async_transfer_pipeline_flush() {
        let pipeline: AsyncTransferPipeline<u8> = AsyncTransferPipeline::new(16);

        for i in 0..8_u8 {
            let data = vec![i; 64];
            pipeline
                .submit(data, TransferDirection::DeviceToHost)
                .expect("submit should succeed");
        }

        // In CPU mode all are already done, but pending_count still tracks them.
        assert_eq!(pipeline.pending_count(), 8);

        pipeline.flush().expect("flush should succeed");

        // After flush, completed transfers should have been drained.
        assert_eq!(pipeline.pending_count(), 0);
    }

    /// Attempting to submit more than `max_pending` incomplete transfers should
    /// return `PipelineFull`.
    ///
    /// Because CPU mode completes transfers synchronously this test uses a
    /// max_pending of 0 to reliably trigger the error.
    #[test]
    fn test_async_transfer_pipeline_full() {
        // A pipeline that never allows any in-flight transfers.
        let pipeline: AsyncTransferPipeline<f32> = AsyncTransferPipeline::new(0);

        let result = pipeline.submit(vec![0.0_f32; 4], TransferDirection::HostToDevice);

        match result {
            Err(AsyncTransferError::PipelineFull(count)) => {
                assert_eq!(count, 0, "should report 0 in-flight when cap is 0");
            }
            other => panic!("expected PipelineFull, got {:?}", other),
        }
    }

    /// DeviceToDevice transfers work the same as host-device transfers in
    /// CPU simulation mode.
    #[test]
    fn test_async_transfer_device_to_device() {
        let pipeline: AsyncTransferPipeline<i32> = AsyncTransferPipeline::new(4);

        let handle = pipeline
            .submit(vec![42_i32; 32], TransferDirection::DeviceToDevice)
            .expect("submit should succeed");

        assert!(handle.is_complete());
    }

    /// Multiple handles should each be independently complete.
    #[test]
    fn test_async_transfer_multiple_handles() {
        let pipeline: AsyncTransferPipeline<f64> = AsyncTransferPipeline::new(16);
        let mut handles = Vec::new();

        for _ in 0..5 {
            let h = pipeline
                .submit(vec![1.0_f64; 8], TransferDirection::HostToDevice)
                .expect("submit should succeed");
            handles.push(h);
        }

        for (i, h) in handles.iter().enumerate() {
            assert!(h.is_complete(), "handle {} should be complete", i);
        }

        assert_eq!(pipeline.in_flight_count(), 0);
    }
}
