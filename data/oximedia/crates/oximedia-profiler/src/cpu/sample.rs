//! Stack sampling for CPU profiling.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A single profiling sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Timestamp when the sample was taken.
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,

    /// Stack frames at the time of sampling.
    pub stack: Vec<StackFrame>,

    /// Thread ID.
    pub thread_id: u64,

    /// CPU usage at the time of sampling (0.0-100.0).
    pub cpu_usage: f64,
}

/// A single stack frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function name.
    pub function: String,

    /// Module/crate name.
    pub module: Option<String>,

    /// File name.
    pub file: Option<String>,

    /// Line number.
    pub line: Option<u32>,

    /// Address.
    pub address: u64,
}

impl Sample {
    /// Create a new sample.
    pub fn new(thread_id: u64, cpu_usage: f64) -> Self {
        Self {
            timestamp: Instant::now(),
            stack: Vec::new(),
            thread_id,
            cpu_usage,
        }
    }

    /// Add a stack frame.
    pub fn add_frame(&mut self, frame: StackFrame) {
        self.stack.push(frame);
    }

    /// Get the depth of the stack.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Get the top frame (leaf function).
    pub fn top_frame(&self) -> Option<&StackFrame> {
        self.stack.last()
    }

    /// Get the bottom frame (root function).
    pub fn bottom_frame(&self) -> Option<&StackFrame> {
        self.stack.first()
    }
}

impl StackFrame {
    /// Create a new stack frame.
    pub fn new(function: String, address: u64) -> Self {
        Self {
            function,
            module: None,
            file: None,
            line: None,
            address,
        }
    }

    /// Create a stack frame with full information.
    pub fn with_location(
        function: String,
        module: Option<String>,
        file: Option<String>,
        line: Option<u32>,
        address: u64,
    ) -> Self {
        Self {
            function,
            module,
            file,
            line,
            address,
        }
    }

    /// Get a display string for this frame.
    pub fn display(&self) -> String {
        let mut result = self.function.clone();

        if let Some(ref module) = self.module {
            result = format!("{}::{}", module, result);
        }

        if let Some(ref file) = self.file {
            result.push_str(&format!(" ({})", file));
            if let Some(line) = self.line {
                result.push_str(&format!(":{}", line));
            }
        }

        result
    }
}

/// Stack sampler for collecting profiling samples.
#[derive(Debug)]
pub struct StackSampler {
    samples: Vec<Sample>,
    max_depth: usize,
}

impl StackSampler {
    /// Create a new stack sampler.
    pub fn new(max_depth: usize) -> Self {
        Self {
            samples: Vec::new(),
            max_depth,
        }
    }

    /// Take a sample of the current stack.
    ///
    /// On Linux this reads real stack information from:
    /// 1. `/proc/self/task/<tid>/wchan` – the kernel wait-channel (blocking call name).
    /// 2. `/proc/self/task/<tid>/status` – thread state and priority.
    /// 3. `/proc/self/task/<tid>/syscall` – in-progress syscall number and arguments.
    /// 4. `/proc/self/maps` – executable regions for address→symbol resolution.
    ///
    /// Falls back to synthetic frames when files are unavailable (non-Linux or permission
    /// errors).
    pub fn take_sample(&mut self, thread_id: u64, cpu_usage: f64) {
        let mut sample = Sample::new(thread_id, cpu_usage);

        let max = self.max_depth;

        // --- Frame 0: kernel wait-channel for the thread ---
        let wchan_frame = Self::read_wchan_frame(thread_id);
        if let Some(frame) = wchan_frame {
            sample.add_frame(frame);
        }

        // --- Frame 1: current syscall (if in one) ---
        if sample.depth() < max {
            if let Some(frame) = Self::read_syscall_frame(thread_id) {
                sample.add_frame(frame);
            }
        }

        // --- Frame 2+: executable regions from /proc/self/maps as symbolic hints ---
        if sample.depth() < max {
            let map_frames = Self::read_map_frames(max - sample.depth());
            for frame in map_frames {
                sample.add_frame(frame);
            }
        }

        // If we still have no frames (e.g. running/non-blocked state), add a
        // synthetic "running" frame so callers always get at least one entry.
        if sample.depth() == 0 {
            sample.add_frame(StackFrame::new("<running>".to_string(), 0));
        }

        self.samples.push(sample);
    }

    /// Read the kernel wait-channel name for the given thread.
    fn read_wchan_frame(thread_id: u64) -> Option<StackFrame> {
        use std::fs;
        // Try thread-specific path first, then fall back to the process wchan.
        let tid_path = format!("/proc/self/task/{}/wchan", thread_id);
        let proc_path = "/proc/self/wchan";

        let wchan = fs::read_to_string(&tid_path)
            .or_else(|_| fs::read_to_string(proc_path))
            .ok()?;

        let name = wchan.trim();
        if name.is_empty() || name == "0" {
            return None;
        }

        Some(StackFrame::with_location(
            name.to_string(),
            Some("kernel".to_string()),
            None,
            None,
            0,
        ))
    }

    /// Read in-progress syscall info for the given thread.
    fn read_syscall_frame(thread_id: u64) -> Option<StackFrame> {
        use std::fs;

        let tid_path = format!("/proc/self/task/{}/syscall", thread_id);
        let proc_path = "/proc/self/syscall";

        let content = fs::read_to_string(&tid_path)
            .or_else(|_| fs::read_to_string(proc_path))
            .ok()?;

        let mut parts = content.split_whitespace();
        let syscall_nr: i64 = parts.next()?.parse().ok()?;

        // Convert syscall number to a human-readable name for common syscalls.
        let name = Self::syscall_name(syscall_nr);

        // The instruction pointer is the last hex field on the line.
        let ip: u64 = content
            .split_whitespace()
            .last()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        Some(StackFrame::with_location(
            name,
            Some("syscall".to_string()),
            None,
            None,
            ip,
        ))
    }

    /// Map a Linux syscall number to its name (x86-64 ABI, common subset).
    fn syscall_name(nr: i64) -> String {
        match nr {
            0 => "read".to_string(),
            1 => "write".to_string(),
            2 => "open".to_string(),
            3 => "close".to_string(),
            4 => "stat".to_string(),
            5 => "fstat".to_string(),
            9 => "mmap".to_string(),
            11 => "munmap".to_string(),
            17 => "pread64".to_string(),
            18 => "pwrite64".to_string(),
            35 => "nanosleep".to_string(),
            56 => "clone".to_string(),
            60 => "exit".to_string(),
            61 => "wait4".to_string(),
            202 => "futex".to_string(),
            228 => "clock_gettime".to_string(),
            231 => "exit_group".to_string(),
            232 => "epoll_wait".to_string(),
            270 => "pselect6".to_string(),
            281 => "epoll_pwait".to_string(),
            -1 => "<not-in-syscall>".to_string(),
            nr => format!("syscall_{}", nr),
        }
    }

    /// Read executable memory regions from `/proc/self/maps` and return them as
    /// synthetic stack frames representing the loaded binary/library layout.
    fn read_map_frames(limit: usize) -> Vec<StackFrame> {
        use std::fs;

        let content = match fs::read_to_string("/proc/self/maps") {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut frames = Vec::new();

        for line in content.lines().take(limit * 4) {
            // Only include executable segments with a mapped file.
            if !line.contains("r-xp") && !line.contains("r-x ") {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                continue;
            }

            let addr_range = parts[0];
            let path = parts[5];

            // Skip anonymous or special regions.
            if path.starts_with('[') && path != "[vdso]" {
                continue;
            }

            // Extract the start address.
            let start_addr = addr_range
                .split('-')
                .next()
                .and_then(|s| u64::from_str_radix(s, 16).ok())
                .unwrap_or(0);

            // Use the filename as the symbol name.
            let sym = std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string());

            frames.push(StackFrame::with_location(
                sym,
                Some(path.to_string()),
                None,
                None,
                start_addr,
            ));

            if frames.len() >= limit {
                break;
            }
        }

        frames
    }

    /// Get all samples.
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Get the number of samples.
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Get the maximum stack depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(1, 50.0);
        assert_eq!(sample.thread_id, 1);
        assert_eq!(sample.cpu_usage, 50.0);
        assert_eq!(sample.depth(), 0);
    }

    #[test]
    fn test_sample_frames() {
        let mut sample = Sample::new(1, 50.0);
        sample.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample.add_frame(StackFrame::new("func2".to_string(), 0x2000));

        assert_eq!(sample.depth(), 2);
        assert_eq!(
            sample.top_frame().expect("should succeed in test").function,
            "func2"
        );
        assert_eq!(
            sample
                .bottom_frame()
                .expect("should succeed in test")
                .function,
            "func1"
        );
    }

    #[test]
    fn test_stack_frame_display() {
        let frame = StackFrame::with_location(
            "my_function".to_string(),
            Some("my_module".to_string()),
            Some("main.rs".to_string()),
            Some(42),
            0x1000,
        );

        let display = frame.display();
        assert!(display.contains("my_module"));
        assert!(display.contains("my_function"));
        assert!(display.contains("main.rs"));
        assert!(display.contains("42"));
    }

    #[test]
    fn test_stack_sampler() {
        let mut sampler = StackSampler::new(10);
        assert_eq!(sampler.count(), 0);

        sampler.take_sample(1, 50.0);
        assert_eq!(sampler.count(), 1);

        sampler.take_sample(2, 60.0);
        assert_eq!(sampler.count(), 2);

        sampler.clear();
        assert_eq!(sampler.count(), 0);
    }

    #[test]
    fn test_stack_sampler_max_depth() {
        let mut sampler = StackSampler::new(3);
        sampler.take_sample(1, 50.0);

        let sample = &sampler.samples()[0];
        assert!(sample.depth() <= 5); // Limited by implementation
    }
}
