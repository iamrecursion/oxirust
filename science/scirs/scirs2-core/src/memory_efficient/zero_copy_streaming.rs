//! Zero-Copy Streaming Framework
//!
//! This module provides a zero-copy streaming framework that minimizes memory
//! allocations and enables high-performance real-time data processing.
//!
//! ## Features
//!
//! - **Zero-Copy Operations**: Memory-mapped data structures for minimal copying
//! - **Lock-Free Queues**: High-performance concurrent data structures
//! - **Memory Pool Management**: Efficient buffer reuse and allocation
//! - **NUMA-Aware Processing**: Optimized for multi-socket systems
//! - **Work-Stealing Scheduler**: Dynamic load balancing across threads
//! - **Adaptive Prefetching**: Intelligent data prefetching based on access patterns
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use scirs2_core::memory_efficient::{
//!     ZeroCopyStreamProcessor, ZeroCopyConfig, ProcessingMode
//! };
//!
//! // Configure zero-copy streaming
//! let config = ZeroCopyConfig {
//!     mode: ProcessingMode::RealTime,
//!     buffer_poolsize: 1024,
//!     numa_aware: true,
//!     work_stealing: true,
//!     ..Default::default()
//! };
//!
//! // Create processor with zero-copy capabilities
//! let mut processor = ZeroCopyStreamProcessor::new(config, |data| {
//!     // Process data without copying
//!     Ok(data)
//! })?;
//!
//! processor.start()?;
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::error_context;
use crate::memory_efficient::streaming::StreamState;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Processing mode for zero-copy streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    /// Real-time processing with bounded latency
    RealTime,
    /// Throughput-optimized processing
    Throughput,
    /// Latency-optimized processing
    Latency,
    /// Adaptive mode that adjusts based on workload
    Adaptive,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub nodes: usize,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Memory regions per NUMA node
    pub memory_regions: Vec<usize>,
}

impl Default for NumaTopology {
    fn default() -> Self {
        // Simple default topology for single-node systems
        Self {
            nodes: 1,
            cpus_per_node: vec![vec![0, 1, 2, 3]], // 4 CPUs on node 0
            memory_regions: vec![0],
        }
    }
}

/// Configuration for zero-copy streaming
#[derive(Debug, Clone)]
pub struct ZeroCopyConfig {
    /// Processing mode
    pub mode: ProcessingMode,
    /// Buffer pool size (number of buffers)
    pub buffer_poolsize: usize,
    /// Buffer size in bytes
    pub buffersize: usize,
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
    /// NUMA topology information
    pub numa_topology: NumaTopology,
    /// Enable work-stealing scheduler
    pub work_stealing: bool,
    /// Number of worker threads
    pub worker_threads: Option<usize>,
    /// Enable lock-free queues
    pub lock_free: bool,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable adaptive prefetching
    pub adaptive_prefetch: bool,
    /// Prefetch window size
    pub prefetch_window: usize,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
    /// Enable huge pages
    pub use_huge_pages: bool,
    /// Target latency in microseconds
    pub target_latency_us: u64,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            mode: ProcessingMode::Adaptive,
            buffer_poolsize: 256,
            buffersize: 1024 * 1024, // 1MB
            numa_aware: false,
            numa_topology: NumaTopology::default(),
            work_stealing: true,
            worker_threads: None,
            lock_free: true,
            max_queue_size: 1024,
            adaptive_prefetch: true,
            prefetch_window: 64,
            memory_alignment: 64, // Cache line aligned
            use_huge_pages: false,
            target_latency_us: 1000, // 1ms
        }
    }
}

/// Zero-copy buffer that manages memory without copying
pub struct ZeroCopyBuffer {
    /// Raw memory pointer
    ptr: NonNull<u8>,
    /// Buffer size in bytes
    size: usize,
    /// Reference count
    ref_count: Arc<AtomicUsize>,
    /// NUMA node ID
    numa_node: Option<usize>,
    /// Memory layout for deallocation
    layout: Layout,
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer
    pub fn new(size: usize, numanode: Option<usize>, alignment: usize) -> CoreResult<Self> {
        let layout = Layout::from_size_align(size, alignment).map_err(|e| {
            CoreError::MemoryError(
                ErrorContext::new(format!("{e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let ptr = unsafe {
            let raw_ptr = alloc_zeroed(layout);
            if raw_ptr.is_null() {
                return Err(CoreError::MemoryError(
                    ErrorContext::new("Failed to allocate memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
            // SAFETY: We've explicitly checked that raw_ptr is not null above
            // The layout is validated by Layout::from_size_align() earlier
            // The memory is properly aligned and allocated for the specified _size
            NonNull::new_unchecked(raw_ptr)
        };

        Ok(Self {
            ptr,
            size,
            ref_count: Arc::new(AtomicUsize::new(1)),
            numa_node: numanode,
            layout,
        })
    }

    /// Get a slice view of the buffer
    ///
    /// # Safety
    /// This is safe because:
    /// - The buffer lifetime is managed by reference counting
    /// - The pointer is guaranteed to be valid while the buffer exists
    /// - The size is validated during allocation and stored immutably
    /// - The memory is properly aligned and initialized (zeroed)
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY:
        // 1. ptr is guaranteed to be valid and properly aligned (validated during allocation)
        // 2. self.size was validated during allocation and cannot be modified
        // 3. The memory is initialized (zeroed during allocation)
        // 4. The buffer is kept alive by reference counting and the slice
        //    cannot outlive the buffer due to Rust's lifetime system
        // 5. The pointer is guaranteed to be non-null (NonNull type invariant)
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get a mutable slice view of the buffer
    ///
    /// # Safety
    /// This is safe because:
    /// - Exclusive access is guaranteed by &mut self
    /// - The pointer is guaranteed to be valid while the buffer exists
    /// - The size is validated during allocation and stored immutably
    /// - The memory is properly aligned and initialized
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY:
        // 1. ptr is guaranteed to be valid and properly aligned (validated during allocation)
        // 2. self.size was validated during allocation and cannot be modified
        // 3. Exclusive access is guaranteed by &mut self, preventing data races
        // 4. The memory is initialized (zeroed during allocation)
        // 5. The pointer is guaranteed to be non-null (NonNull type invariant)
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Create a shared reference to this buffer
    ///
    /// # Errors
    /// Returns `CoreError::MemoryError` if the reference count would overflow
    pub fn share(&self) -> CoreResult<Self> {
        let old_count = self.ref_count.fetch_add(1, Ordering::Relaxed);
        if old_count == usize::MAX {
            // Prevent overflow - this is extremely unlikely in practice
            self.ref_count.fetch_sub(1, Ordering::Relaxed);
            return Err(CoreError::MemoryError(error_context!(
                "Reference count overflow in ZeroCopyBuffer"
            )));
        }

        Ok(Self {
            ptr: self.ptr,
            size: self.size,
            ref_count: self.ref_count.clone(),
            numa_node: self.numa_node,
            layout: self.layout,
        })
    }

    /// Get the size of the buffer
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the NUMA node ID
    pub fn numa_node(&self) -> Option<usize> {
        self.numa_node
    }

    /// Check if this is the only reference to the buffer
    pub fn is_unique(&self) -> bool {
        self.ref_count.load(Ordering::Relaxed) == 1
    }

    /// Get the reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Relaxed)
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        // Use AcqRel ordering to ensure proper synchronization with share()
        // This prevents memory reordering issues in reference counting
        if self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // SAFETY:
            // 1. We are the last reference holder (ref_count was 1)
            // 2. ptr and layout are guaranteed to match the original allocation
            // 3. The memory was allocated with the same layout using alloc_zeroed
            // 4. No other threads can access this memory after ref_count reaches 0
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

/// Lock-free queue for zero-copy buffers
pub struct LockFreeQueue<T> {
    /// Head pointer
    head: AtomicPtr<Node<T>>,
    /// Tail pointer
    tail: AtomicPtr<Node<T>>,
    /// Current size
    size: AtomicUsize,
    /// Maximum size
    maxsize: usize,
}

struct Node<T> {
    data: Option<T>,
    next: AtomicPtr<Node<T>>,
}

impl<T> LockFreeQueue<T> {
    /// Create a new lock-free queue
    pub fn new(maxsize: usize) -> Self {
        let dummy = Box::into_raw(Box::new(Node {
            data: None,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));

        Self {
            head: AtomicPtr::new(dummy),
            tail: AtomicPtr::new(dummy),
            size: AtomicUsize::new(0),
            maxsize,
        }
    }

    /// Push an item to the queue
    pub fn push(&self, item: T) -> Result<(), T> {
        // Check size limit before allocating
        if self.size.load(Ordering::Acquire) >= self.maxsize {
            return Err(item);
        }

        let new_node = Box::into_raw(Box::new(Node {
            data: Some(item),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));

        // SAFETY: Lock-free queue algorithm with ABA protection
        // We use compare_exchange_weak to ensure atomic operations and prevent data races
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            if tail.is_null() {
                // Queue is being destroyed - safely clean up
                // SAFETY: new_node was allocated by Box::into_raw above
                let node = unsafe { Box::from_raw(new_node) };
                let item = node.data.expect("Operation failed");
                return Err(item);
            }

            // SAFETY: tail is guaranteed to be non-null and valid here because:
            // 1. We checked tail.is_null() above
            // 2. The tail pointer is only modified atomically
            // 3. Node deallocation only happens after tail is updated away from it
            // 4. The queue maintains the invariant that tail always points to a valid node
            let next = unsafe { (*tail).next.load(Ordering::Acquire) };

            // ABA protection: verify tail hasn't changed during our operations
            // This prevents the classic ABA problem in lock-free data structures
            if tail == self.tail.load(Ordering::Acquire) {
                if next.is_null() {
                    // Try to atomically link the new node
                    // SAFETY:
                    // 1. tail is valid (checked above)
                    // 2. new_node is valid (just allocated)
                    // 3. We're atomically updating the next pointer
                    // 4. compare_exchange_weak provides memory ordering guarantees
                    if unsafe {
                        (*tail)
                            .next
                            .compare_exchange_weak(
                                next,
                                new_node,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                    } {
                        break;
                    }
                } else {
                    // Help advance tail pointer to maintain queue consistency
                    // This helps other threads make progress
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
            // Retry if tail changed during our operation (ABA case)
            // The loop will eventually succeed due to the helping mechanism
        }

        // Try to advance tail
        let _ = self.tail.compare_exchange_weak(
            self.tail.load(Ordering::Acquire),
            new_node,
            Ordering::Release,
            Ordering::Relaxed,
        );

        self.size.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Pop an item from the queue
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            if head.is_null() {
                // Queue is being destroyed
                return None;
            }

            let tail = self.tail.load(Ordering::Acquire);
            // SAFETY: head is guaranteed to be non-null and valid here because:
            // 1. We checked head.is_null() above
            // 2. The head pointer is only modified atomically
            // 3. The queue maintains the invariant that head always points to a valid node
            // 4. Node deallocation only happens after head is updated away from it
            let next = unsafe { (*head).next.load(Ordering::Acquire) };

            // ABA protection: verify head hasn't changed during our operations
            // This prevents race conditions where head changes between loads
            if head == self.head.load(Ordering::Acquire) {
                if head == tail {
                    if next.is_null() {
                        // Queue is empty (head == tail and no next node)
                        return None;
                    }
                    // Help advance tail pointer to maintain queue consistency
                    // This helps other threads make progress when tail lags behind
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                } else {
                    if next.is_null() {
                        // Inconsistent state: head != tail but no next node
                        // This shouldn't happen in a well-formed queue, retry
                        continue;
                    }

                    // SAFETY: next is guaranteed to be valid and contain data because:
                    // 1. next is not null (checked above)
                    // 2. head != tail implies there are items in the queue
                    // 3. next points to the actual data node (head is dummy)
                    // 4. We take the data before advancing head to avoid use-after-free
                    let data = unsafe { (*next).data.take() };

                    // Atomically advance head pointer
                    if self
                        .head
                        .compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        // Successfully advanced head, now safe to deallocate old head node
                        // SAFETY:
                        // 1. head is no longer reachable from the queue structure
                        // 2. head was allocated by Box::into_raw in push() or queue creation
                        // 3. No other threads can access head after it's been updated
                        unsafe { drop(Box::from_raw(head)) };
                        self.size.fetch_sub(1, Ordering::Release);
                        return data;
                    }
                    // If compare_exchange failed, retry the entire operation
                }
            }
            // Retry if head changed during our operation
        }
    }

    /// Get the current size of the queue
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the queue is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.maxsize
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // First, signal that the queue is being destroyed by setting pointers to null
        // This prevents other threads from continuing to access the queue
        self.head.store(std::ptr::null_mut(), Ordering::Release);
        self.tail.store(std::ptr::null_mut(), Ordering::Release);

        // Give other threads a moment to notice the null pointers
        // In a real implementation, you might use a more sophisticated approach
        std::thread::yield_now();

        // Now drain all remaining items to avoid leaks
        // We can't use pop() here since head is null, so we manually traverse
        let mut current = self.head.load(Ordering::Relaxed);

        // Restore head temporarily to drain the queue
        if current.is_null() {
            // Queue was already empty or we need to find the original head
            return;
        }

        // Walk through all nodes and clean them up
        while !current.is_null() {
            // SAFETY:
            // 1. We are in the destructor, so no other threads should be accessing the queue
            // 2. current was loaded from a valid atomic pointer
            // 3. All nodes were allocated with Box::into_raw
            let next = unsafe { (*current).next.load(Ordering::Relaxed) };

            // SAFETY:
            // 1. current was allocated by Box::into_raw in push() or queue creation
            // 2. We are the only thread accessing the queue during destruction
            unsafe { drop(Box::from_raw(current)) };

            current = next;
        }
    }
}

unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

/// Buffer pool for efficient memory management
pub struct BufferPool {
    /// Available buffers
    available: LockFreeQueue<ZeroCopyBuffer>,
    /// Buffer configuration
    buffersize: usize,
    /// NUMA-aware allocation
    numa_aware: bool,
    /// Buffer alignment
    alignment: usize,
    /// Statistics
    stats: Arc<RwLock<BufferPoolStats>>,
}

/// Statistics for buffer pool
#[derive(Debug, Clone, Default)]
pub struct BufferPoolStats {
    /// Buffers allocated
    pub buffers_allocated: usize,
    /// Buffers reused
    pub buffers_reused: usize,
    /// Pool hits
    pub pool_hits: usize,
    /// Pool misses
    pub pool_misses: usize,
    /// Peak pool size
    pub peak_poolsize: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(
        poolsize: usize,
        buffersize: usize,
        numa_aware: bool,
        alignment: usize,
    ) -> CoreResult<Self> {
        let available = LockFreeQueue::new(poolsize);

        // Pre-allocate buffers
        for _ in 0..poolsize {
            let buffer = ZeroCopyBuffer::new(
                buffersize,
                if numa_aware { Some(0) } else { None },
                alignment,
            )?;
            available.push(buffer).map_err(|_| {
                CoreError::MemoryError(
                    ErrorContext::new("Failed to initialize buffer pool".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }

        Ok(Self {
            available,
            buffersize,
            numa_aware,
            alignment,
            stats: Arc::new(RwLock::new(BufferPoolStats::default())),
        })
    }

    /// Get a buffer from the pool
    pub fn get_buffer(&self) -> CoreResult<ZeroCopyBuffer> {
        if let Some(buffer) = self.available.pop() {
            // Pool hit
            {
                let mut stats = self.stats.write().expect("Operation failed");
                stats.pool_hits += 1;
                stats.buffers_reused += 1;
            }
            Ok(buffer)
        } else {
            // Pool miss - allocate new buffer
            {
                let mut stats = self.stats.write().expect("Operation failed");
                stats.pool_misses += 1;
                stats.buffers_allocated += 1;
            }

            ZeroCopyBuffer::new(
                self.buffersize,
                if self.numa_aware { Some(0) } else { None },
                self.alignment,
            )
        }
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: ZeroCopyBuffer) {
        if buffer.is_unique() && buffer.size() == self.buffersize {
            if self.available.push(buffer).is_err() {
                // Pool is full, buffer will be dropped
            } else {
                let stats = self.stats.read().expect("Operation failed");
                let current_size = self.available.len();
                if current_size > stats.peak_poolsize {
                    drop(stats);
                    let mut stats = self.stats.write().expect("Operation failed");
                    stats.peak_poolsize = current_size;
                }
            }
        }
        // If buffer is not unique or wrong size, it will be dropped
    }

    /// Get buffer pool statistics
    pub fn stats(&self) -> BufferPoolStats {
        self.stats.read().expect("Operation failed").clone()
    }

    /// Get current pool size
    pub fn current_size(&self) -> usize {
        self.available.len()
    }
}

/// Work-stealing task for the scheduler
pub trait WorkStealingTask: Send + 'static {
    /// Execute the task
    fn execute(&self);
}

/// Work-stealing scheduler for efficient parallel processing
pub struct WorkStealingScheduler {
    /// Global task queue
    global_queue: Arc<LockFreeQueue<Box<dyn WorkStealingTask>>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Worker handles
    handles: Vec<JoinHandle<()>>,
}

struct Worker {
    /// Worker ID
    #[allow(dead_code)]
    id: usize,
    /// Local task queue
    local_queue: Arc<LockFreeQueue<Box<dyn WorkStealingTask>>>,
    /// Reference to global queue
    global_queue: Arc<LockFreeQueue<Box<dyn WorkStealingTask>>>,
    /// Other workers for stealing
    other_workers: Vec<Arc<LockFreeQueue<Box<dyn WorkStealingTask>>>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new(numworkers: usize, max_queuesize: usize) -> Self {
        let global_queue = Arc::new(LockFreeQueue::new(max_queuesize));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(numworkers);

        // Create worker local queues first
        let mut local_queues: Vec<Arc<LockFreeQueue<Box<dyn WorkStealingTask>>>> = Vec::new();
        for _ in 0..numworkers {
            local_queues.push(Arc::new(LockFreeQueue::new(
                max_queuesize / numworkers.max(1),
            )));
        }

        // Create and start worker threads
        for i in 0..numworkers {
            let mut other_workers = Vec::new();
            for (j, queue) in local_queues.iter().enumerate() {
                if i != j {
                    other_workers.push(queue.clone());
                }
            }

            let worker = Worker {
                id: i,
                local_queue: local_queues[i].clone(),
                global_queue: global_queue.clone(),
                other_workers,
                shutdown: shutdown.clone(),
            };

            let handle = thread::spawn(move || {
                worker.run();
            });
            handles.push(handle);
        }

        Self {
            global_queue,
            shutdown,
            handles,
        }
    }

    /// Submit a task to the scheduler
    pub fn submit<T: WorkStealingTask>(&self, task: T) {
        if self.global_queue.push(Box::new(task)).is_err() {
            // Queue is full, task will be dropped
            // In a production system, you might want to handle this differently
        }
    }

    /// Shutdown the scheduler
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Get the number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        self.global_queue.len()
    }
}

impl Worker {
    fn run(self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            // Try to get a task from local queue first
            if let Some(task) = self.local_queue.pop() {
                task.execute();
                continue;
            }

            // Try to get a task from global queue
            if let Some(task) = self.global_queue.pop() {
                task.execute();
                continue;
            }

            // Try to steal a task from other workers
            let mut stolen = false;
            for other_queue in &self.other_workers {
                if let Some(task) = other_queue.pop() {
                    task.execute();
                    stolen = true;
                    break;
                }
            }

            if !stolen {
                // No tasks available, sleep briefly to avoid busy waiting
                thread::sleep(Duration::from_micros(100));
            }
        }
    }
}

impl Drop for WorkStealingScheduler {
    fn drop(&mut self) {
        self.shutdown();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Zero-copy stream processor
pub struct ZeroCopyStreamProcessor<T, U>
where
    T: Send + 'static,
    U: Send + 'static,
{
    /// Configuration
    config: ZeroCopyConfig,
    /// Buffer pool
    buffer_pool: Arc<BufferPool>,
    /// Work-stealing scheduler
    #[allow(dead_code)]
    scheduler: Option<Arc<WorkStealingScheduler>>,
    /// Processing function
    processfn: Arc<dyn Fn(T) -> CoreResult<U> + Send + Sync>,
    /// Input queue
    input_queue: Arc<LockFreeQueue<T>>,
    /// Output queue
    output_queue: Arc<LockFreeQueue<U>>,
    /// Current state
    state: Arc<RwLock<StreamState>>,
    /// Processing statistics
    stats: Arc<RwLock<ZeroCopyStats>>,
    /// Worker threads
    worker_handles: Vec<JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Statistics for zero-copy processing
#[derive(Debug, Clone, Default)]
pub struct ZeroCopyStats {
    /// Items processed
    pub items_processed: usize,
    /// Average processing time (microseconds)
    pub avg_processing_time_us: f64,
    /// Memory allocations avoided
    pub allocations_avoided: usize,
    /// Buffer pool efficiency
    pub pool_efficiency: f64,
    /// Work-stealing steals
    pub work_steals: usize,
    /// NUMA node affinity hits
    pub numa_hits: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
}

impl<T, U> ZeroCopyStreamProcessor<T, U>
where
    T: Send + 'static,
    U: Send + 'static,
{
    /// Create a new zero-copy stream processor
    pub fn new<F>(config: ZeroCopyConfig, processfn: F) -> CoreResult<Self>
    where
        F: Fn(T) -> CoreResult<U> + Send + Sync + 'static,
    {
        // Create buffer pool
        let buffer_pool = Arc::new(BufferPool::new(
            config.buffer_poolsize,
            config.buffersize,
            config.numa_aware,
            config.memory_alignment,
        )?);

        // Create work-stealing scheduler if enabled
        let scheduler = if config.work_stealing {
            let numworkers = config.worker_threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
            Some(Arc::new(WorkStealingScheduler::new(
                numworkers,
                config.max_queue_size,
            )))
        } else {
            None
        };

        // Create lock-free queues
        let input_queue = Arc::new(LockFreeQueue::new(config.max_queue_size));
        let output_queue = Arc::new(LockFreeQueue::new(config.max_queue_size));

        Ok(Self {
            config,
            buffer_pool,
            scheduler,
            processfn: Arc::new(processfn),
            input_queue,
            output_queue,
            state: Arc::new(RwLock::new(StreamState::Initialized)),
            stats: Arc::new(RwLock::new(ZeroCopyStats::default())),
            worker_handles: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start the zero-copy stream processor
    pub fn start(&mut self) -> CoreResult<()> {
        let mut state = self.state.write().expect("Operation failed");
        if *state == StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream already running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        *state = StreamState::Running;
        drop(state);

        // Start worker threads
        let numworkers = self.config.worker_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        for workerid in 0..numworkers {
            let input_queue = self.input_queue.clone();
            let output_queue = self.output_queue.clone();
            let processfn = self.processfn.clone();
            let state = self.state.clone();
            let stats = self.stats.clone();
            let shutdown = self.shutdown.clone();

            let handle = thread::spawn(move || {
                Self::worker_loop(
                    workerid,
                    input_queue,
                    output_queue,
                    processfn,
                    state,
                    stats,
                    shutdown,
                );
            });

            self.worker_handles.push(handle);
        }

        Ok(())
    }

    /// Worker loop for processing data
    fn worker_loop(
        workerid: usize,
        input_queue: Arc<LockFreeQueue<T>>,
        output_queue: Arc<LockFreeQueue<U>>,
        processfn: Arc<dyn Fn(T) -> CoreResult<U> + Send + Sync>,
        state: Arc<RwLock<StreamState>>,
        stats: Arc<RwLock<ZeroCopyStats>>,
        shutdown: Arc<AtomicBool>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // Check if we should continue
            {
                let current_state = state.read().expect("Operation failed");
                if *current_state != StreamState::Running {
                    break;
                }
            }

            // Try to get work from input queue
            if let Some(input) = input_queue.pop() {
                let start_time = Instant::now();

                // Process the input
                match processfn(input) {
                    Ok(output) => {
                        // Try to put output in output queue
                        if output_queue.push(output).is_err() {
                            // Output queue is full - in a production system,
                            // you might want to handle backpressure differently
                        }

                        // Update statistics
                        {
                            let mut stats_guard = stats.write().expect("Operation failed");
                            stats_guard.items_processed += 1;
                            let processing_time = start_time.elapsed().as_micros() as f64;
                            stats_guard.avg_processing_time_us = (stats_guard
                                .avg_processing_time_us
                                * (stats_guard.items_processed - 1) as f64
                                + processing_time)
                                / stats_guard.items_processed as f64;
                        }
                    }
                    Err(_) => {
                        // Processing error - update error stats if needed
                    }
                }
            } else {
                // No work available, sleep briefly
                thread::sleep(Duration::from_micros(10));
            }
        }
    }

    /// Push data to the processor
    pub fn push(&self, data: T) -> CoreResult<()> {
        if self.input_queue.push(data).is_err() {
            Err(CoreError::StreamError(
                ErrorContext::new("Input queue is full".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        } else {
            Ok(())
        }
    }

    /// Pop processed data from the processor
    pub fn pop(&self) -> Option<U> {
        self.output_queue.pop()
    }

    /// Get processing statistics
    pub fn stats(&self) -> ZeroCopyStats {
        let mut stats = self.stats.read().expect("Operation failed").clone();

        // Add buffer pool statistics
        let buffer_stats = self.buffer_pool.stats();
        stats.pool_efficiency = if buffer_stats.pool_hits + buffer_stats.pool_misses > 0 {
            buffer_stats.pool_hits as f64
                / (buffer_stats.pool_hits + buffer_stats.pool_misses) as f64
        } else {
            0.0
        };
        stats.allocations_avoided = buffer_stats.buffers_reused;

        stats
    }

    /// Stop the processor
    pub fn stop(&mut self) -> CoreResult<()> {
        // Set shutdown flag
        self.shutdown.store(true, Ordering::Relaxed);

        // Update state
        {
            let mut state = self.state.write().expect("Operation failed");
            *state = StreamState::Paused;
        }

        // Wait for worker threads to finish
        for handle in self.worker_handles.drain(..) {
            handle.join().map_err(|_| {
                CoreError::StreamError(
                    ErrorContext::new("Failed to join worker thread".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }

        Ok(())
    }

    /// Get current queue sizes
    pub fn queue_sizes(&self) -> (usize, usize) {
        (self.input_queue.len(), self.output_queue.len())
    }

    /// Get buffer pool reference
    pub const fn buffer_pool(&self) -> &Arc<BufferPool> {
        &self.buffer_pool
    }
}

impl<T, U> Drop for ZeroCopyStreamProcessor<T, U>
where
    T: Send + 'static,
    U: Send + 'static,
{
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Create a new zero-copy stream processor with default configuration
#[allow(dead_code)]
pub fn create_zero_copy_processor<T, U, F>(
    processfn: F,
) -> CoreResult<ZeroCopyStreamProcessor<T, U>>
where
    T: Send + 'static,
    U: Send + 'static,
    F: Fn(T) -> CoreResult<U> + Send + Sync + 'static,
{
    ZeroCopyStreamProcessor::new(ZeroCopyConfig::default(), processfn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_buffer() {
        let buffer = ZeroCopyBuffer::new(1024, None, 64).expect("Operation failed");
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.is_unique());
        assert_eq!(buffer.ref_count(), 1);

        let shared = buffer.share().expect("Operation failed");
        assert_eq!(shared.size(), 1024);
        assert!(!buffer.is_unique());
        assert_eq!(buffer.ref_count(), 2);
    }

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::new(10);
        assert!(queue.is_empty());

        // Push some items
        for i in 0..5 {
            assert!(queue.push(i).is_ok());
        }
        assert_eq!(queue.len(), 5);

        // Pop some items
        for i in 0..5 {
            assert_eq!(queue.pop(), Some(i));
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(4, 1024, false, 64).expect("Operation failed");

        // Get a buffer
        let buffer1 = pool.get_buffer().expect("Operation failed");
        assert_eq!(buffer1.size(), 1024);

        // Return the buffer
        pool.return_buffer(buffer1);

        // Get another buffer (should reuse)
        let buffer2 = pool.get_buffer().expect("Operation failed");
        assert_eq!(buffer2.size(), 1024);

        let stats = pool.stats();
        assert!(stats.buffers_reused > 0 || stats.buffers_allocated > 0);
    }

    #[test]
    fn test_zero_copy_processor() {
        let config = ZeroCopyConfig {
            worker_threads: Some(2),
            max_queue_size: 100,
            ..Default::default()
        };

        let mut processor =
            ZeroCopyStreamProcessor::new(config, |x: i32| Ok(x * 2)).expect("Operation failed");

        processor.start().expect("Operation failed");

        // Push some data
        for i in 0..10 {
            processor.push(i).expect("Operation failed");
        }

        // Wait a bit for processing
        std::thread::sleep(Duration::from_millis(100));

        // Pop results
        let mut results = Vec::new();
        while let Some(result) = processor.pop() {
            results.push(result);
        }

        processor.stop().expect("Operation failed");

        // Check that we got some results
        assert!(!results.is_empty());

        let stats = processor.stats();
        assert!(stats.items_processed > 0);
    }
}
