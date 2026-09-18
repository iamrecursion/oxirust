//! GPU kernel configuration and launch result types.

/// Configuration for launching a GPU kernel.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Name of the kernel (for diagnostics and profiling).
    pub name: String,
    /// Grid dimensions (x, y, z) in number of thread blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z) in number of threads per block.
    pub block_dim: (u32, u32, u32),
    /// Amount of dynamic shared memory per block in bytes.
    pub shared_memory_bytes: usize,
    /// Optional stream identifier for asynchronous execution.
    pub stream_id: Option<usize>,
}

impl KernelConfig {
    /// Create a new `KernelConfig` with default dimensions:
    /// - grid: (1, 1, 1)
    /// - block: (32, 1, 1)
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_memory_bytes: 0,
            stream_id: None,
        }
    }

    /// Set the grid dimensions (builder pattern).
    pub fn with_grid(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_dim = (x, y, z);
        self
    }

    /// Set the block dimensions (builder pattern).
    pub fn with_block(mut self, x: u32, y: u32, z: u32) -> Self {
        self.block_dim = (x, y, z);
        self
    }

    /// Set the shared memory size in bytes (builder pattern).
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory_bytes = bytes;
        self
    }

    /// Returns the total number of threads across all blocks.
    pub fn total_threads(&self) -> u32 {
        let grid_total = self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2;
        let block_total = self.block_dim.0 * self.block_dim.1 * self.block_dim.2;
        grid_total * block_total
    }

    /// Calculate the number of blocks required to process `n_elements` elements
    /// with `threads_per_block` threads per block (ceiling division).
    pub fn blocks_needed(n_elements: u32, threads_per_block: u32) -> u32 {
        if threads_per_block == 0 {
            return 0;
        }
        n_elements.div_ceil(threads_per_block)
    }
}

/// Result produced after a GPU kernel is launched (or stubbed).
#[derive(Debug, Clone)]
pub struct KernelLaunchResult {
    /// Name of the kernel that was launched.
    pub kernel_name: String,
    /// Elapsed time in microseconds, if timing was available.
    pub elapsed_us: Option<u64>,
    /// Grid dimensions used for the launch.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions used for the launch.
    pub block_dim: (u32, u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_new() {
        let config = KernelConfig::new("elementwise_add");
        assert_eq!(config.name, "elementwise_add");
        assert_eq!(config.grid_dim, (1, 1, 1));
        assert_eq!(config.block_dim, (32, 1, 1));
        assert_eq!(config.shared_memory_bytes, 0);
        assert!(config.stream_id.is_none());
    }

    #[test]
    fn test_kernel_config_builder() {
        let config = KernelConfig::new("matmul")
            .with_grid(64, 64, 1)
            .with_block(16, 16, 1)
            .with_shared_memory(2048);
        assert_eq!(config.grid_dim, (64, 64, 1));
        assert_eq!(config.block_dim, (16, 16, 1));
        assert_eq!(config.shared_memory_bytes, 2048);
    }

    #[test]
    fn test_total_threads() {
        let config = KernelConfig::new("test")
            .with_grid(4, 2, 1)
            .with_block(32, 1, 1);
        // 4*2*1 = 8 blocks, 32*1*1 = 32 threads/block → 256 total
        assert_eq!(config.total_threads(), 256);
    }

    #[test]
    fn test_blocks_needed() {
        assert_eq!(KernelConfig::blocks_needed(1024, 32), 32);
        assert_eq!(KernelConfig::blocks_needed(1025, 32), 33);
        assert_eq!(KernelConfig::blocks_needed(0, 32), 0);
        assert_eq!(KernelConfig::blocks_needed(32, 32), 1);
        // Edge: zero threads_per_block should not panic.
        assert_eq!(KernelConfig::blocks_needed(100, 0), 0);
    }

    #[test]
    fn test_kernel_launch_result() {
        let result = KernelLaunchResult {
            kernel_name: "test_kernel".to_string(),
            elapsed_us: Some(42),
            grid_dim: (8, 1, 1),
            block_dim: (128, 1, 1),
        };
        assert_eq!(result.kernel_name, "test_kernel");
        assert_eq!(result.elapsed_us, Some(42));
        assert_eq!(result.grid_dim, (8, 1, 1));
        assert_eq!(result.block_dim, (128, 1, 1));
    }
}
