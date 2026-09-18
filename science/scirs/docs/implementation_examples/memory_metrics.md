# Enhanced Memory Management with Metrics Collection

This document outlines a comprehensive implementation plan for extending the memory management module in `scirs2-core` with detailed metrics collection and reporting capabilities.

## Overview

Enhanced memory metrics will provide critical insights into memory usage patterns, allocation frequencies, and potential bottlenecks in scientific computing workloads. This system will allow developers to identify memory-intensive operations, track peak usage, and optimize memory-critical applications.

## Architecture

```
scirs2-core/src/memory/
├── mod.rs                  # Module exports
├── buffer_pool.rs          # Existing buffer pool implementation
├── chunk_processor.rs      # Existing chunk processor
├── zero_copy.rs            # Existing zero-copy views
├── metrics/                # New metrics submodule
│   ├── mod.rs
│   ├── collector.rs        # Metrics collection system
│   ├── aggregator.rs       # Metrics aggregation
│   ├── reporter.rs         # Reporting utilities
│   ├── visualizer.rs       # Optional visualization tools
│   └── event.rs            # Memory events definition
└── tracked/                # Memory-tracked variants
    ├── mod.rs              
    ├── buffer_pool.rs      # Tracked buffer pool
    ├── chunk_processor.rs  # Tracked chunk processor
    └── allocator.rs        # Custom allocator with tracking
```

## Core Components

### 1. Memory Event System

```rust
#[derive(Debug, Clone)]
pub enum MemoryEventType {
    Allocation,
    Deallocation,
    Resize,
    Access,
    Transfer,
}

#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Event type
    pub event_type: MemoryEventType,
    /// Component that generated the event (e.g., "BufferPool", "ChunkProcessor")
    pub component: String,
    /// Optional operation context (e.g., "matrix_multiply", "fft")
    pub context: Option<String>,
    /// Allocation size in bytes
    pub size: usize,
    /// Memory address (for correlation)
    pub address: usize,
    /// Thread ID
    pub thread_id: u64,
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Call stack (if enabled)
    pub call_stack: Option<Vec<String>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryEvent {
    pub fn new(
        event_type: MemoryEventType,
        component: impl Into<String>,
        size: usize,
        address: usize,
    ) -> Self {
        Self {
            event_type,
            component: component.into(),
            context: None,
            size,
            address,
            thread_id: std::thread::current().id().as_u64(),
            timestamp: std::time::Instant::now(),
            call_stack: None,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
    
    pub fn with_call_stack(mut self) -> Self {
        if cfg!(feature = "memory_call_stack") {
            self.call_stack = Some(capture_call_stack(3)); // Skip 3 frames
        }
        self
    }
    
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Capture a call stack (simplified implementation)
#[cfg(feature = "memory_call_stack")]
fn capture_call_stack(skip_frames: usize) -> Vec<String> {
    // Use backtrace crate to capture call stack
    let backtrace = backtrace::Backtrace::new();
    backtrace.frames()
        .iter()
        .skip(skip_frames)
        .take(10) // Limit to 10 frames
        .filter_map(|frame| {
            frame.symbols().get(0).map(|symbol| {
                format!("{}", symbol.name().unwrap_or_else(|| backtrace::SymbolName::new(&[])))
            })
        })
        .collect()
}

#[cfg(not(feature = "memory_call_stack"))]
fn capture_call_stack(_skip_frames: usize) -> Vec<String> {
    Vec::new()
}
```

### 2. Metrics Collector

```rust
pub struct MemoryMetricsConfig {
    /// Whether to collect events
    pub enabled: bool,
    /// Whether to capture call stacks (requires memory_call_stack feature)
    pub capture_call_stacks: bool,
    /// Maximum number of events to store
    pub max_events: usize,
    /// Whether to aggregate events in real-time
    pub real_time_aggregation: bool,
    /// Event sampling rate (1.0 = all events, 0.1 = 10% of events)
    pub sampling_rate: f64,
}

impl Default for MemoryMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capture_call_stacks: cfg!(feature = "memory_call_stack"),
            max_events: 10000,
            real_time_aggregation: true,
            sampling_rate: 1.0,
        }
    }
}

pub struct MemoryMetricsCollector {
    /// Configuration
    config: MemoryMetricsConfig,
    /// Collected events (if not aggregated)
    events: RwLock<VecDeque<MemoryEvent>>,
    /// Current memory usage by component
    current_usage: RwLock<HashMap<String, usize>>,
    /// Peak memory usage by component
    peak_usage: RwLock<HashMap<String, usize>>,
    /// Allocation count by component
    allocation_count: RwLock<HashMap<String, usize>>,
    /// Total allocated bytes by component
    total_allocated: RwLock<HashMap<String, usize>>,
    /// Average allocation size by component
    avg_allocation_size: RwLock<HashMap<String, f64>>,
    /// Start time
    start_time: std::time::Instant,
    /// Random number generator for sampling
    rng: Mutex<Random>,
}

impl MemoryMetricsCollector {
    pub fn new(config: MemoryMetricsConfig) -> Self {
        Self {
            config,
            events: RwLock::new(VecDeque::with_capacity(1000)),
            current_usage: RwLock::new(HashMap::new()),
            peak_usage: RwLock::new(HashMap::new()),
            allocation_count: RwLock::new(HashMap::new()),
            total_allocated: RwLock::new(HashMap::new()),
            avg_allocation_size: RwLock::new(HashMap::new()),
            start_time: std::time::Instant::now(),
            rng: Mutex::new(Random::default()),
        }
    }
    
    /// Record a memory event
    pub fn record_event(&self, event: MemoryEvent) {
        if !self.config.enabled {
            return;
        }
        
        // Sample events if sampling rate < 1.0
        if self.config.sampling_rate < 1.0 {
            let mut rng = self.rng.lock().unwrap();
            if rng.random_range(0.0..1.0) > self.config.sampling_rate {
                return;
            }
        }
        
        // Update aggregated metrics in real-time if enabled
        if self.config.real_time_aggregation {
            self.update_metrics(&event);
        }
        
        // Store the event if we're keeping raw events
        if self.config.max_events > 0 {
            let mut events = self.events.write().unwrap();
            events.push_back(event);
            
            // Limit the number of stored events
            while events.len() > self.config.max_events {
                events.pop_front();
            }
        }
    }
    
    /// Update aggregated metrics based on an event
    fn update_metrics(&self, event: &MemoryEvent) {
        match event.event_type {
            MemoryEventType::Allocation => {
                // Update current usage
                let mut current_usage = self.current_usage.write().unwrap();
                let component_usage = current_usage.entry(event.component.clone()).or_insert(0);
                *component_usage += event.size;
                
                // Update peak usage if current > peak
                let mut peak_usage = self.peak_usage.write().unwrap();
                let peak = peak_usage.entry(event.component.clone()).or_insert(0);
                *peak = (*peak).max(*component_usage);
                
                // Update allocation count
                let mut allocation_count = self.allocation_count.write().unwrap();
                let count = allocation_count.entry(event.component.clone()).or_insert(0);
                *count += 1;
                
                // Update total allocated
                let mut total_allocated = self.total_allocated.write().unwrap();
                let total = total_allocated.entry(event.component.clone()).or_insert(0);
                *total += event.size;
                
                // Update average allocation size
                let mut avg_allocation_size = self.avg_allocation_size.write().unwrap();
                let avg = avg_allocation_size.entry(event.component.clone()).or_insert(0.0);
                *avg = (*avg * (*count - 1) as f64 + event.size as f64) / *count as f64;
            },
            MemoryEventType::Deallocation => {
                // Update current usage
                let mut current_usage = self.current_usage.write().unwrap();
                let component_usage = current_usage.entry(event.component.clone()).or_insert(0);
                *component_usage = component_usage.saturating_sub(event.size);
            },
            MemoryEventType::Resize => {
                // Handle resize events (could be positive or negative change)
                if let Some(old_size) = event.metadata.get("old_size").and_then(|s| s.parse::<usize>().ok()) {
                    let size_diff = event.size as isize - old_size as isize;
                    
                    let mut current_usage = self.current_usage.write().unwrap();
                    let component_usage = current_usage.entry(event.component.clone()).or_insert(0);
                    
                    if size_diff > 0 {
                        *component_usage += size_diff as usize;
                    } else {
                        *component_usage = component_usage.saturating_sub((-size_diff) as usize);
                    }
                    
                    // Update peak usage if needed
                    let mut peak_usage = self.peak_usage.write().unwrap();
                    let peak = peak_usage.entry(event.component.clone()).or_insert(0);
                    *peak = (*peak).max(*component_usage);
                }
            },
            _ => {
                // Other event types don't affect metrics
            }
        }
    }
    
    /// Get the current memory usage for a specific component
    pub fn get_current_usage(&self, component: &str) -> usize {
        let current_usage = self.current_usage.read().unwrap();
        *current_usage.get(component).unwrap_or(&0)
    }
    
    /// Get the peak memory usage for a specific component
    pub fn get_peak_usage(&self, component: &str) -> usize {
        let peak_usage = self.peak_usage.read().unwrap();
        *peak_usage.get(component).unwrap_or(&0)
    }
    
    /// Get total memory usage across all components
    pub fn get_total_current_usage(&self) -> usize {
        let current_usage = self.current_usage.read().unwrap();
        current_usage.values().sum()
    }
    
    /// Get peak memory usage across all components
    pub fn get_total_peak_usage(&self) -> usize {
        let current_usage = self.current_usage.read().unwrap();
        let peak_usage = self.peak_usage.read().unwrap();
        
        // Either sum of component peaks or peak of total current usage
        let component_sum: usize = peak_usage.values().sum();
        let peak_total = current_usage.values().sum::<usize>();
        
        component_sum.max(peak_total)
    }
    
    /// Get allocation statistics for a component
    pub fn get_allocation_stats(&self, component: &str) -> Option<AllocationStats> {
        let allocation_count = self.allocation_count.read().unwrap();
        let count = *allocation_count.get(component)?;
        
        let total_allocated = self.total_allocated.read().unwrap();
        let total = *total_allocated.get(component)?;
        
        let avg_allocation_size = self.avg_allocation_size.read().unwrap();
        let avg = *avg_allocation_size.get(component)?;
        
        let peak_usage = self.peak_usage.read().unwrap();
        let peak = *peak_usage.get(component)?;
        
        Some(AllocationStats {
            count,
            total_bytes: total,
            average_size: avg,
            peak_usage: peak,
        })
    }
    
    /// Generate a memory report
    pub fn generate_report(&self) -> MemoryReport {
        let current_usage = self.current_usage.read().unwrap();
        let peak_usage = self.peak_usage.read().unwrap();
        let allocation_count = self.allocation_count.read().unwrap();
        let total_allocated = self.total_allocated.read().unwrap();
        let avg_allocation_size = self.avg_allocation_size.read().unwrap();
        
        let mut components = HashSet::new();
        components.extend(current_usage.keys().cloned());
        components.extend(peak_usage.keys().cloned());
        components.extend(allocation_count.keys().cloned());
        
        let mut component_stats = HashMap::new();
        for component in components {
            let stats = ComponentMemoryStats {
                current_usage: *current_usage.get(&component).unwrap_or(&0),
                peak_usage: *peak_usage.get(&component).unwrap_or(&0),
                allocation_count: *allocation_count.get(&component).unwrap_or(&0),
                total_allocated: *total_allocated.get(&component).unwrap_or(&0),
                avg_allocation_size: *avg_allocation_size.get(&component).unwrap_or(&0.0),
            };
            
            component_stats.insert(component, stats);
        }
        
        MemoryReport {
            total_current_usage: current_usage.values().sum(),
            total_peak_usage: self.get_total_peak_usage(),
            total_allocation_count: allocation_count.values().sum(),
            total_allocated_bytes: total_allocated.values().sum(),
            component_stats,
            duration: self.start_time.elapsed(),
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        let mut events = self.events.write().unwrap();
        events.clear();
        
        let mut current_usage = self.current_usage.write().unwrap();
        current_usage.clear();
        
        let mut peak_usage = self.peak_usage.write().unwrap();
        peak_usage.clear();
        
        let mut allocation_count = self.allocation_count.write().unwrap();
        allocation_count.clear();
        
        let mut total_allocated = self.total_allocated.write().unwrap();
        total_allocated.clear();
        
        let mut avg_allocation_size = self.avg_allocation_size.write().unwrap();
        avg_allocation_size.clear();
        
        self.start_time = std::time::Instant::now();
    }
}

/// Global metrics collector instance
pub fn global_metrics_collector() -> &'static MemoryMetricsCollector {
    static INSTANCE: OnceCell<MemoryMetricsCollector> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        MemoryMetricsCollector::new(MemoryMetricsConfig::default())
    })
}
```

### 3. Memory Statistics and Reporting

```rust
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub count: usize,
    pub total_bytes: usize,
    pub average_size: f64,
    pub peak_usage: usize,
}

#[derive(Debug, Clone)]
pub struct ComponentMemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub total_allocated: usize,
    pub avg_allocation_size: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub total_current_usage: usize,
    pub total_peak_usage: usize,
    pub total_allocation_count: usize,
    pub total_allocated_bytes: usize,
    pub component_stats: HashMap<String, ComponentMemoryStats>,
    pub duration: Duration,
}

impl MemoryReport {
    /// Format the report as a string
    pub fn format(&self) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("Memory Report (duration: {:.2?})\n", self.duration));
        output.push_str(&format!("Total Current Usage: {}\n", format_bytes(self.total_current_usage)));
        output.push_str(&format!("Total Peak Usage: {}\n", format_bytes(self.total_peak_usage)));
        output.push_str(&format!("Total Allocations: {}\n", self.total_allocation_count));
        output.push_str(&format!("Total Allocated Bytes: {}\n", format_bytes(self.total_allocated_bytes)));
        
        output.push_str("\nComponent Statistics:\n");
        
        // Sort components by peak usage (descending)
        let mut components: Vec<_> = self.component_stats.iter().collect();
        components.sort_by(|a, b| b.1.peak_usage.cmp(&a.1.peak_usage));
        
        for (component, stats) in components {
            output.push_str(&format!("\n  {}\n", component));
            output.push_str(&format!("    Current Usage: {}\n", format_bytes(stats.current_usage)));
            output.push_str(&format!("    Peak Usage: {}\n", format_bytes(stats.peak_usage)));
            output.push_str(&format!("    Allocation Count: {}\n", stats.allocation_count));
            output.push_str(&format!("    Total Allocated: {}\n", format_bytes(stats.total_allocated)));
            output.push_str(&format!("    Avg Allocation Size: {}\n", format_bytes(stats.avg_allocation_size as usize)));
            
            // Calculate reuse ratio
            if stats.total_allocated > 0 {
                let reuse_ratio = stats.total_allocated as f64 / stats.peak_usage as f64;
                output.push_str(&format!("    Memory Reuse Ratio: {:.2}\n", reuse_ratio));
            }
            
            // Calculate allocation frequency
            if self.duration.as_secs_f64() > 0.0 {
                let alloc_per_sec = stats.allocation_count as f64 / self.duration.as_secs_f64();
                output.push_str(&format!("    Allocations/sec: {:.2}\n", alloc_per_sec));
            }
        }
        
        output
    }
    
    /// Export the report as JSON
    pub fn to_json(&self) -> serde_json::Value {
        let mut component_stats = serde_json::Map::new();
        for (component, stats) in &self.component_stats {
            let stats_obj = serde_json::json!({
                "current_usage": stats.current_usage,
                "peak_usage": stats.peak_usage,
                "allocation_count": stats.allocation_count,
                "total_allocated": stats.total_allocated,
                "avg_allocation_size": stats.avg_allocation_size,
                "reuse_ratio": if stats.peak_usage > 0 { stats.total_allocated as f64 / stats.peak_usage as f64 } else { 0.0 },
                "alloc_per_sec": if self.duration.as_secs_f64() > 0.0 { stats.allocation_count as f64 / self.duration.as_secs_f64() } else { 0.0 },
            });
            component_stats.insert(component.clone(), stats_obj);
        }
        
        serde_json::json!({
            "total_current_usage": self.total_current_usage,
            "total_peak_usage": self.total_peak_usage,
            "total_allocation_count": self.total_allocation_count,
            "total_allocated_bytes": self.total_allocated_bytes,
            "duration_seconds": self.duration.as_secs_f64(),
            "components": component_stats,
        })
    }
}

/// Format bytes in human-readable format
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
```

### 4. Tracked Buffer Pool

```rust
pub struct TrackedBufferPool<T: Clone + Default> {
    component_name: String,
    context: Option<String>,
    inner_pool: BufferPool<T>,
}

impl<T: Clone + Default> TrackedBufferPool<T> {
    pub fn new(component_name: impl Into<String>) -> Self {
        Self {
            component_name: component_name.into(),
            context: None,
            inner_pool: BufferPool::new(),
        }
    }
    
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
    
    pub fn acquire_vec(&mut self, capacity: usize) -> Vec<T> {
        // Use the inner pool to get a vector
        let vec = self.inner_pool.acquire_vec(capacity);
        
        // Record the allocation
        let event = MemoryEvent::new(
            MemoryEventType::Allocation,
            &self.component_name,
            std::mem::size_of::<T>() * capacity,
            vec.as_ptr() as usize,
        )
        .with_call_stack();
        
        // Add context if available
        let event = if let Some(context) = &self.context {
            event.with_context(context)
        } else {
            event
        };
        
        // Add metadata
        let event = event
            .with_metadata("element_type", std::any::type_name::<T>())
            .with_metadata("element_size", std::mem::size_of::<T>().to_string())
            .with_metadata("capacity", capacity.to_string());
        
        // Record the event
        global_metrics_collector().record_event(event);
        
        vec
    }
    
    pub fn release_vec(&mut self, vec: Vec<T>) {
        // Record the deallocation
        let event = MemoryEvent::new(
            MemoryEventType::Deallocation,
            &self.component_name,
            std::mem::size_of::<T>() * vec.capacity(),
            vec.as_ptr() as usize,
        )
        .with_call_stack();
        
        // Add context if available
        let event = if let Some(context) = &self.context {
            event.with_context(context)
        } else {
            event
        };
        
        // Add metadata
        let event = event
            .with_metadata("element_type", std::any::type_name::<T>())
            .with_metadata("element_size", std::mem::size_of::<T>().to_string())
            .with_metadata("capacity", vec.capacity().to_string());
        
        // Record the event
        global_metrics_collector().record_event(event);
        
        // Release the vector back to the pool
        self.inner_pool.release_vec(vec);
    }
    
    // Other methods from BufferPool...
}
```

### 5. Tracked Chunk Processor

```rust
pub struct TrackedChunkProcessor {
    component_name: String,
    context: Option<String>,
    inner_processor: ChunkProcessor,
}

impl TrackedChunkProcessor {
    pub fn new(component_name: impl Into<String>) -> Self {
        Self {
            component_name: component_name.into(),
            context: None,
            inner_processor: ChunkProcessor::new(),
        }
    }
    
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
    
    pub fn with_chunk_size(mut self, chunk_size: (usize, usize)) -> Self {
        self.inner_processor = self.inner_processor.with_chunk_size(chunk_size);
        self
    }
    
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.inner_processor = self.inner_processor.with_overlap(overlap);
        self
    }
    
    pub fn process_array2<T, F>(
        &self,
        array: &Array2<T>,
        operation: F,
    ) -> Array2<T>
    where
        T: Clone,
        F: Fn(&Array2<T>) -> Array2<T>,
    {
        // Start tracking metrics for this operation
        let start_time = std::time::Instant::now();
        let array_size = std::mem::size_of::<T>() * array.len();
        
        // Record the allocation event for the input array (virtual, as we don't actually allocate it)
        let allocation_event = MemoryEvent::new(
            MemoryEventType::Access,
            &self.component_name,
            array_size,
            array.as_ptr() as usize,
        )
        .with_call_stack();
        
        // Add context if available
        let allocation_event = if let Some(context) = &self.context {
            allocation_event.with_context(context)
        } else {
            allocation_event
        };
        
        // Add metadata
        let allocation_event = allocation_event
            .with_metadata("operation", "process_array2")
            .with_metadata("element_type", std::any::type_name::<T>())
            .with_metadata("element_size", std::mem::size_of::<T>().to_string())
            .with_metadata("shape", format!("{:?}", array.shape()));
        
        // Record the event
        global_metrics_collector().record_event(allocation_event);
        
        // Process the array with the inner processor
        let result = self.inner_processor.process_array2(array, |chunk| {
            // Record chunk processing event
            let chunk_event = MemoryEvent::new(
                MemoryEventType::Access,
                &self.component_name,
                std::mem::size_of::<T>() * chunk.len(),
                chunk.as_ptr() as usize,
            )
            .with_call_stack();
            
            // Add context if available
            let chunk_event = if let Some(context) = &self.context {
                chunk_event.with_context(context)
            } else {
                chunk_event
            };
            
            // Add metadata
            let chunk_event = chunk_event
                .with_metadata("operation", "process_chunk")
                .with_metadata("element_type", std::any::type_name::<T>())
                .with_metadata("element_size", std::mem::size_of::<T>().to_string())
                .with_metadata("shape", format!("{:?}", chunk.shape()));
            
            // Record the event
            global_metrics_collector().record_event(chunk_event);
            
            // Execute the operation on the chunk
            operation(chunk)
        });
        
        // Record metrics for the result array
        let result_size = std::mem::size_of::<T>() * result.len();
        
        let result_event = MemoryEvent::new(
            MemoryEventType::Allocation,
            &self.component_name,
            result_size,
            result.as_ptr() as usize,
        )
        .with_call_stack();
        
        // Add context if available
        let result_event = if let Some(context) = &self.context {
            result_event.with_context(context)
        } else {
            result_event
        };
        
        // Add metadata
        let result_event = result_event
            .with_metadata("operation", "process_array2_result")
            .with_metadata("element_type", std::any::type_name::<T>())
            .with_metadata("element_size", std::mem::size_of::<T>().to_string())
            .with_metadata("shape", format!("{:?}", result.shape()))
            .with_metadata("duration_ms", format!("{}", start_time.elapsed().as_millis()));
        
        // Record the event
        global_metrics_collector().record_event(result_event);
        
        result
    }
    
    // Other methods from ChunkProcessor...
}
```

### 6. Memory Metrics Integration API

```rust
/// Memory metrics API for easy integration
pub struct MemoryMetrics;

impl MemoryMetrics {
    /// Start tracking memory for a scope
    pub fn track_scope(component: &str, context: Option<&str>) -> ScopeTracker {
        ScopeTracker::new(component, context)
    }
    
    /// Track an allocation
    pub fn track_allocation<T>(component: &str, ptr: *const T, count: usize) {
        if !global_metrics_collector().config.enabled {
            return;
        }
        
        let size = std::mem::size_of::<T>() * count;
        
        let event = MemoryEvent::new(
            MemoryEventType::Allocation,
            component,
            size,
            ptr as usize,
        )
        .with_call_stack()
        .with_metadata("element_type", std::any::type_name::<T>())
        .with_metadata("element_size", std::mem::size_of::<T>().to_string())
        .with_metadata("count", count.to_string());
        
        global_metrics_collector().record_event(event);
    }
    
    /// Track a deallocation
    pub fn track_deallocation<T>(component: &str, ptr: *const T, count: usize) {
        if !global_metrics_collector().config.enabled {
            return;
        }
        
        let size = std::mem::size_of::<T>() * count;
        
        let event = MemoryEvent::new(
            MemoryEventType::Deallocation,
            component,
            size,
            ptr as usize,
        )
        .with_call_stack()
        .with_metadata("element_type", std::any::type_name::<T>())
        .with_metadata("element_size", std::mem::size_of::<T>().to_string())
        .with_metadata("count", count.to_string());
        
        global_metrics_collector().record_event(event);
    }
    
    /// Get the current memory report
    pub fn get_report() -> MemoryReport {
        global_metrics_collector().generate_report()
    }
    
    /// Reset all metrics
    pub fn reset() {
        global_metrics_collector().reset();
    }
    
    /// Get current memory usage
    pub fn current_usage() -> usize {
        global_metrics_collector().get_total_current_usage()
    }
    
    /// Get peak memory usage
    pub fn peak_usage() -> usize {
        global_metrics_collector().get_total_peak_usage()
    }
    
    /// Configure the memory metrics system
    pub fn configure(config: MemoryMetricsConfig) {
        // Replace the global collector with a new one using the specified config
        // This is a simplification; in reality we'd need a better approach to update the global instance
        // or use thread-local configuration
    }
}

/// Track memory usage within a scope
pub struct ScopeTracker {
    component: String,
    context: Option<String>,
    start_time: std::time::Instant,
    start_usage: usize,
}

impl ScopeTracker {
    pub fn new(component: &str, context: Option<&str>) -> Self {
        let component = component.to_string();
        let context = context.map(|s| s.to_string());
        let start_time = std::time::Instant::now();
        
        let start_usage = global_metrics_collector().get_current_usage(&component);
        
        Self {
            component,
            context,
            start_time,
            start_usage,
        }
    }
}

impl Drop for ScopeTracker {
    fn drop(&mut self) {
        if !global_metrics_collector().config.enabled {
            return;
        }
        
        let end_usage = global_metrics_collector().get_current_usage(&self.component);
        let duration = self.start_time.elapsed();
        
        // Record a summary event for the scope
        let event = MemoryEvent::new(
            MemoryEventType::Transfer, // Using Transfer as a generic event type
            &self.component,
            end_usage,
            0, // No specific address
        )
        .with_call_stack();
        
        // Add context if available
        let event = if let Some(context) = &self.context {
            event.with_context(context)
        } else {
            event
        };
        
        // Add metadata
        let event = event
            .with_metadata("operation", "scope_summary")
            .with_metadata("start_usage", self.start_usage.to_string())
            .with_metadata("end_usage", end_usage.to_string())
            .with_metadata("delta", (end_usage as isize - self.start_usage as isize).to_string())
            .with_metadata("duration_ms", duration.as_millis().to_string());
        
        global_metrics_collector().record_event(event);
    }
}
```

## Usage Examples

### Basic Usage

```rust
use scirs2_core::memory::metrics::{MemoryMetrics, TrackedBufferPool};
use ndarray::Array2;

fn process_large_matrix() -> Array2<f64> {
    // Start tracking this scope
    let _tracker = MemoryMetrics::track_scope("matrix_processing", Some("process_large_matrix"));
    
    // Create a tracked buffer pool
    let mut pool = TrackedBufferPool::<f64>::new("matrix_buffer_pool")
        .with_context("process_large_matrix");
    
    // Use the pool
    let mut buffer = pool.acquire_vec(1_000_000);
    
    // Fill the buffer with data
    for i in 0..buffer.len() {
        buffer[i] = i as f64;
    }
    
    // Create a result array
    let mut result = Array2::<f64>::zeros((1000, 1000));
    
    // Track this allocation
    MemoryMetrics::track_allocation("result_array", result.as_ptr(), result.len());
    
    // Process the data
    for i in 0..1000 {
        for j in 0..1000 {
            result[[i, j]] = buffer[i * 1000 + j];
        }
    }
    
    // Release the buffer back to the pool
    pool.release_vec(buffer);
    
    // Print memory usage stats
    println!("Current memory usage: {}", MemoryMetrics::current_usage());
    println!("Peak memory usage: {}", MemoryMetrics::peak_usage());
    
    result
}
```

### Using Tracked Chunk Processor

```rust
use scirs2_core::memory::metrics::{MemoryMetrics, TrackedChunkProcessor};
use ndarray::Array2;

fn process_image(image: &Array2<f32>) -> Array2<f32> {
    // Create a tracked chunk processor
    let processor = TrackedChunkProcessor::new("image_processor")
        .with_context("gaussian_blur")
        .with_chunk_size((512, 512))
        .with_overlap(5);
    
    // Process the image in chunks
    let result = processor.process_array2(image, |chunk| {
        // Apply a Gaussian blur to the chunk
        apply_gaussian_blur(chunk, 2.0)
    });
    
    // Get and print the memory report
    let report = MemoryMetrics::get_report();
    println!("{}", report.format());
    
    result
}

fn apply_gaussian_blur(chunk: &Array2<f32>, sigma: f32) -> Array2<f32> {
    // Implementation of Gaussian blur
    // ...
    
    // For this example, just return a clone of the chunk
    chunk.clone()
}
```

### Memory Report Example

```rust
fn run_benchmark() {
    // Reset metrics
    MemoryMetrics::reset();
    
    // Run the benchmark
    benchmark_matrix_operations();
    
    // Get the report
    let report = MemoryMetrics::get_report();
    
    // Print the report
    println!("{}", report.format());
    
    // Export as JSON
    let json = report.to_json();
    std::fs::write("memory_report.json", json.to_string()).unwrap();
    
    // Optionally create a visualization
    #[cfg(feature = "memory_visualization")]
    {
        use scirs2_core::memory::metrics::Visualizer;
        Visualizer::from_report(&report)
            .plot_usage_over_time()
            .save("memory_usage.png")
            .unwrap();
    }
}
```

## Integration with Other Core Features

### Integration with Profiling

```rust
use scirs2_core::memory::metrics::MemoryMetrics;
use scirs2_core::profiling::{Timer, Profiler};

fn process_with_profiling_and_memory_tracking() {
    // Reset metrics
    MemoryMetrics::reset();
    
    // Start profiling
    let _timer = Timer::start("process_with_profiling_and_memory_tracking");
    
    // Track memory for this scope
    let _mem_tracker = MemoryMetrics::track_scope("main_processing", None);
    
    // Do some memory-intensive operations
    let start_mem = MemoryMetrics::current_usage();
    
    // Record both timing and memory usage
    {
        let _op_timer = Timer::start("matrix_creation");
        let _op_mem = MemoryMetrics::track_scope("matrix_creation", None);
        
        // Create a large matrix
        let matrix = vec![0.0; 10_000_000];
        MemoryMetrics::track_allocation("matrix", matrix.as_ptr(), matrix.len());
        
        // Process the matrix
        // ...
    }
    
    // Get both profiling and memory reports
    let timing_report = Profiler::global().lock().unwrap().generate_report();
    let memory_report = MemoryMetrics::get_report();
    
    println!("Timing Report:\n{}", timing_report);
    println!("\nMemory Report:\n{}", memory_report.format());
    
    // Calculate memory growth
    let end_mem = MemoryMetrics::current_usage();
    let growth = end_mem as isize - start_mem as isize;
    
    println!("Memory growth: {} bytes", growth);
}
```

### Integration with GPU Acceleration

```rust
use scirs2_core::memory::metrics::MemoryMetrics;
use scirs2_core::gpu::{GpuContext, GpuBackend};

fn gpu_memory_tracking() {
    // Create a GPU context
    let ctx = GpuContext::new(GpuBackend::preferred()).unwrap();
    
    // Track GPU memory allocation
    let _tracker = MemoryMetrics::track_scope("gpu_operations", Some("matrix_multiply"));
    
    // Allocate memory on GPU
    let buffer_size = 1_000_000;
    let buffer = ctx.create_buffer::<f32>(buffer_size);
    
    // Track the GPU buffer allocation
    MemoryMetrics::track_allocation(
        "gpu_buffer", 
        &buffer as *const _ as *const u8, 
        std::mem::size_of::<f32>() * buffer_size
    );
    
    // Use the buffer
    // ...
    
    // Track deallocation (this would normally happen automatically)
    MemoryMetrics::track_deallocation(
        "gpu_buffer", 
        &buffer as *const _ as *const u8, 
        std::mem::size_of::<f32>() * buffer_size
    );
    
    // Generate report
    let report = MemoryMetrics::get_report();
    println!("{}", report.format());
}
```

## Benefits

1. **Memory Optimization**: Identify memory usage patterns and optimize algorithms accordingly
2. **Leak Detection**: Find memory leaks through detailed allocation/deallocation tracking
3. **Performance Insights**: Correlate memory usage with performance metrics
4. **Resource Planning**: Better understand memory requirements for large datasets
5. **Visualization**: Create visual representations of memory usage patterns
6. **Debugging**: Diagnose memory-related issues with detailed information

## Future Enhancements

1. **Memory Pressure Simulation**: Test how algorithms behave under memory-constrained conditions
2. **Call-stack Based Analysis**: Identify memory-intensive callsites and suggest optimizations
3. **Automatic Memory Optimization**: Dynamically adjust chunk sizes and buffer pools based on usage patterns
4. **Cross-device Memory Tracking**: Track memory usage across CPU, GPU, and other devices
5. **Memory Visualization**: Create interactive visualizations of memory usage over time
6. **Anomaly Detection**: Automatically detect unusual memory patterns and potential issues