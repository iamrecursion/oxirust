//! Plugin sandboxing: permission enforcement, resource limits, and timeout checking.
//!
//! This module provides a pure-Rust, no-OS-syscall sandbox model.  Policy is
//! enforced by checking bitmask flags, path allow-lists, and atomic counters
//! before each resource acquisition.  Actual OS-level isolation (e.g. seccomp,
//! namespaces) is outside the scope of this crate and should be layered on top
//! if required.
//!
//! # Permission Flags
//!
//! | Constant              | Value | Meaning                            |
//! |-----------------------|-------|------------------------------------|
//! | `PERM_NETWORK`        | 0x01  | Access external network             |
//! | `PERM_FILESYSTEM`     | 0x02  | Read/write the local filesystem     |
//! | `PERM_GPU`            | 0x04  | Submit work to a GPU                |
//! | `PERM_AUDIO`          | 0x08  | Access audio hardware               |
//! | `PERM_VIDEO`          | 0x10  | Access video capture hardware       |
//! | `PERM_MEMORY_LARGE`   | 0x20  | Allocate > `max_memory_mb` of RAM   |
//!
//! # Fine-Grained Filesystem Restrictions
//!
//! When `PERM_FILESYSTEM` is granted, the plugin may further be constrained to
//! a list of explicitly allowed paths.  Access to any path not in the allow-list
//! is denied even if `PERM_FILESYSTEM` is set.  If the allow-list is empty,
//! the entire filesystem is permitted (subject to the OS).
//!
//! # Resource Usage Tracking
//!
//! [`SandboxContext`] records both memory allocation and simulated CPU-time
//! via atomic counters.  Callers report CPU work in nanoseconds using
//! [`SandboxContext::charge_cpu_ns`]; the context compares the accumulated
//! total against `max_cpu_ns` and returns [`SandboxError::CpuExceeded`] when
//! the limit is breached.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

// ── Permission flags ──────────────────────────────────────────────────────────

/// Permission to open network connections.
pub const PERM_NETWORK: u32 = 0x01;
/// Permission to read or write the local filesystem.
pub const PERM_FILESYSTEM: u32 = 0x02;
/// Permission to use a GPU.
pub const PERM_GPU: u32 = 0x04;
/// Permission to access audio hardware.
pub const PERM_AUDIO: u32 = 0x08;
/// Permission to access video capture hardware.
pub const PERM_VIDEO: u32 = 0x10;
/// Permission to allocate large amounts of memory (above `max_memory_mb`).
pub const PERM_MEMORY_LARGE: u32 = 0x20;

// ── PermissionSet ─────────────────────────────────────────────────────────────

/// A bitmask-based set of granted permissions, optionally combined with
/// fine-grained filesystem path restrictions.
///
/// Operations return `Self` by value for builder-pattern chaining.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionSet {
    bits: u32,
    /// Allow-list of filesystem paths the plugin may access.
    ///
    /// When non-empty and `PERM_FILESYSTEM` is set, only these paths (and
    /// their descendants) are accessible.  An empty list means "all paths
    /// permitted" (legacy behaviour).
    allowed_paths: Vec<PathBuf>,
}

impl PermissionSet {
    /// Create a set with no permissions granted.
    pub fn new() -> Self {
        Self {
            bits: 0,
            allowed_paths: Vec::new(),
        }
    }

    /// Create a set with all known permissions granted (no path restrictions).
    pub fn with_all() -> Self {
        Self {
            bits: PERM_NETWORK
                | PERM_FILESYSTEM
                | PERM_GPU
                | PERM_AUDIO
                | PERM_VIDEO
                | PERM_MEMORY_LARGE,
            allowed_paths: Vec::new(),
        }
    }

    /// Grant the given permission flag(s), returning the updated set.
    pub fn grant(self, flag: u32) -> Self {
        Self {
            bits: self.bits | flag,
            allowed_paths: self.allowed_paths,
        }
    }

    /// Revoke the given permission flag(s), returning the updated set.
    pub fn revoke(self, flag: u32) -> Self {
        Self {
            bits: self.bits & !flag,
            allowed_paths: self.allowed_paths,
        }
    }

    /// Test whether the given permission flag(s) are all granted.
    pub fn has(&self, flag: u32) -> bool {
        self.bits & flag == flag
    }

    /// Return the raw bitmask.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Add a path to the filesystem allow-list.
    ///
    /// When the allow-list is non-empty, only paths that are equal to or
    /// are descendants of an allowed path may be accessed (requires
    /// `PERM_FILESYSTEM` to be set as well).
    ///
    /// The path is stored as-is (no canonicalisation is performed here;
    /// canonicalise before calling if required).
    #[must_use]
    pub fn allow_path(mut self, path: impl Into<PathBuf>) -> Self {
        let p = path.into();
        if !self.allowed_paths.contains(&p) {
            self.allowed_paths.push(p);
        }
        self
    }

    /// Remove a path from the filesystem allow-list.
    #[must_use]
    pub fn deny_path(mut self, path: &Path) -> Self {
        self.allowed_paths.retain(|p| p.as_path() != path);
        self
    }

    /// Return the filesystem path allow-list.
    pub fn allowed_paths(&self) -> &[PathBuf] {
        &self.allowed_paths
    }

    /// Check whether `path` is permitted by the current allow-list rules.
    ///
    /// Returns `true` if:
    /// - `PERM_FILESYSTEM` is not set → always false (filesystem denied), or
    /// - `PERM_FILESYSTEM` is set AND the allow-list is empty → always true, or
    /// - `PERM_FILESYSTEM` is set AND `path` starts with any allowed path entry.
    pub fn is_path_allowed(&self, path: &Path) -> bool {
        if !self.has(PERM_FILESYSTEM) {
            return false;
        }
        if self.allowed_paths.is_empty() {
            return true;
        }
        self.allowed_paths
            .iter()
            .any(|allowed| path.starts_with(allowed))
    }
}

impl Default for PermissionSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── SandboxConfig ─────────────────────────────────────────────────────────────

/// Static configuration for a plugin sandbox.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Which permissions the plugin is allowed to exercise.
    pub permissions: PermissionSet,
    /// Maximum RSS memory the plugin may consume (MiB).
    pub max_memory_mb: usize,
    /// Maximum CPU time as a percentage of one core (0–100).
    pub max_cpu_percent: u8,
    /// Wall-clock timeout for a single plugin operation (milliseconds).
    pub timeout_ms: u64,
    /// Maximum cumulative simulated CPU time (nanoseconds, 0 = unlimited).
    ///
    /// When non-zero, calls to [`SandboxContext::charge_cpu_ns`] will fail
    /// once this budget is exhausted.
    pub max_cpu_ns: u64,
}

impl Default for SandboxConfig {
    /// Restrictive defaults: no permissions, 256 MiB, 50 % CPU, 5-second timeout.
    fn default() -> Self {
        Self {
            permissions: PermissionSet::new(),
            max_memory_mb: 256,
            max_cpu_percent: 50,
            timeout_ms: 5_000,
            max_cpu_ns: 0, // unlimited by default
        }
    }
}

impl SandboxConfig {
    /// Create a fully permissive configuration (useful for testing).
    pub fn permissive() -> Self {
        Self {
            permissions: PermissionSet::with_all(),
            max_memory_mb: usize::MAX / (1024 * 1024),
            max_cpu_percent: 100,
            timeout_ms: u64::MAX,
            max_cpu_ns: 0, // unlimited
        }
    }
}

// ── SandboxError ──────────────────────────────────────────────────────────────

/// Errors that arise when a plugin violates its sandbox policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SandboxError {
    /// The plugin attempted to exercise a permission it does not hold.
    #[error("permission denied: requested 0x{requested:02X}, available 0x{available:02X}")]
    PermissionDenied {
        /// The flag(s) requested.
        requested: u32,
        /// The flags currently granted.
        available: u32,
    },

    /// The plugin attempted to access a filesystem path not in its allow-list.
    #[error("filesystem path denied: {path}")]
    PathDenied {
        /// The path the plugin tried to access.
        path: String,
    },

    /// The plugin attempted to allocate more memory than its limit allows.
    #[error("memory limit exceeded: used {used} bytes, limit {limit} bytes")]
    MemoryExceeded {
        /// Current allocation (bytes).
        used: usize,
        /// Configured limit (bytes).
        limit: usize,
    },

    /// The plugin exceeded its wall-clock timeout.
    #[error("timeout: elapsed {elapsed_ms} ms")]
    Timeout {
        /// Elapsed milliseconds since the context was created.
        elapsed_ms: u64,
    },

    /// The plugin exceeded its CPU quota.
    #[error("CPU quota exceeded")]
    CpuExceeded,
}

// ── ResourceSnapshot ──────────────────────────────────────────────────────────

/// A point-in-time snapshot of a plugin's resource usage.
#[derive(Debug, Clone, Copy)]
pub struct ResourceSnapshot {
    /// Memory currently allocated by the plugin (bytes).
    pub memory_bytes: usize,
    /// Cumulative simulated CPU time charged to this context (nanoseconds).
    pub cpu_ns: u64,
    /// Wall-clock time elapsed since context creation (milliseconds).
    pub elapsed_ms: u64,
}

// ── SandboxContext ────────────────────────────────────────────────────────────

/// A live sandbox context that enforces resource policy checks.
///
/// Each plugin execution should be associated with a `SandboxContext`.
/// The context tracks memory consumption via an atomic counter and records
/// its start time for timeout enforcement.
///
/// # Resource Tracking
///
/// - Memory is tracked via [`check_memory`](Self::check_memory) /
///   [`release_memory`](Self::release_memory).
/// - CPU time can be charged via [`charge_cpu_ns`](Self::charge_cpu_ns).
/// - A live snapshot is available via [`resource_snapshot`](Self::resource_snapshot).
pub struct SandboxContext {
    /// Policy governing this context.
    pub config: SandboxConfig,
    /// Atomically tracked memory usage (bytes).
    used_memory: AtomicUsize,
    /// Accumulated simulated CPU time (nanoseconds).
    used_cpu_ns: AtomicU64,
    /// Moment this context was created (used for timeout checks).
    start_time: Instant,
}

impl SandboxContext {
    /// Create a new context from a `SandboxConfig`.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            used_memory: AtomicUsize::new(0),
            used_cpu_ns: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Verify that the given permission flag(s) are granted.
    ///
    /// # Errors
    /// Returns [`SandboxError::PermissionDenied`] if any bit in `flag` is not granted.
    pub fn check_permission(&self, flag: u32) -> Result<(), SandboxError> {
        if !self.config.permissions.has(flag) {
            Err(SandboxError::PermissionDenied {
                requested: flag,
                available: self.config.permissions.bits(),
            })
        } else {
            Ok(())
        }
    }

    /// Verify that the plugin is allowed to access the given filesystem `path`.
    ///
    /// Checks both the `PERM_FILESYSTEM` bit and the path allow-list.
    ///
    /// # Errors
    /// Returns [`SandboxError::PermissionDenied`] if `PERM_FILESYSTEM` is not set,
    /// or [`SandboxError::PathDenied`] if the path is not in the allow-list.
    pub fn check_path(&self, path: &Path) -> Result<(), SandboxError> {
        if !self.config.permissions.has(PERM_FILESYSTEM) {
            return Err(SandboxError::PermissionDenied {
                requested: PERM_FILESYSTEM,
                available: self.config.permissions.bits(),
            });
        }
        if !self.config.permissions.is_path_allowed(path) {
            return Err(SandboxError::PathDenied {
                path: path.to_string_lossy().into_owned(),
            });
        }
        Ok(())
    }

    /// Attempt to record a memory allocation of `requested` bytes.
    ///
    /// Atomically increments the used-memory counter; if the result exceeds
    /// the configured limit the increment is rolled back and an error is returned.
    ///
    /// # Errors
    /// Returns [`SandboxError::MemoryExceeded`] if the limit would be breached.
    pub fn check_memory(&self, requested: usize) -> Result<(), SandboxError> {
        let limit_bytes = self.config.max_memory_mb.saturating_mul(1024 * 1024);
        let prev = self.used_memory.fetch_add(requested, Ordering::Relaxed);
        let new_total = prev.saturating_add(requested);
        if new_total > limit_bytes {
            // Roll back.
            self.used_memory.fetch_sub(requested, Ordering::Relaxed);
            Err(SandboxError::MemoryExceeded {
                used: new_total,
                limit: limit_bytes,
            })
        } else {
            Ok(())
        }
    }

    /// Release memory that was previously recorded with `check_memory`.
    pub fn release_memory(&self, bytes: usize) {
        self.used_memory.fetch_sub(
            bytes.min(self.used_memory.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }

    /// Charge `ns` nanoseconds of simulated CPU time to this context.
    ///
    /// If `max_cpu_ns` in the configuration is non-zero and the accumulated
    /// total would exceed it, the charge is rolled back and
    /// [`SandboxError::CpuExceeded`] is returned.
    ///
    /// # Errors
    /// Returns [`SandboxError::CpuExceeded`] if the CPU budget is exhausted.
    pub fn charge_cpu_ns(&self, ns: u64) -> Result<(), SandboxError> {
        let limit = self.config.max_cpu_ns;
        if limit == 0 {
            // Unlimited — just record the charge.
            self.used_cpu_ns.fetch_add(ns, Ordering::Relaxed);
            return Ok(());
        }
        let prev = self.used_cpu_ns.fetch_add(ns, Ordering::Relaxed);
        if prev.saturating_add(ns) > limit {
            // Roll back.
            self.used_cpu_ns.fetch_sub(ns, Ordering::Relaxed);
            Err(SandboxError::CpuExceeded)
        } else {
            Ok(())
        }
    }

    /// Check whether the configured wall-clock timeout has been exceeded.
    ///
    /// # Errors
    /// Returns [`SandboxError::Timeout`] if the elapsed time exceeds `timeout_ms`.
    pub fn check_timeout(&self) -> Result<(), SandboxError> {
        let elapsed = self.start_time.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        if elapsed_ms > self.config.timeout_ms {
            Err(SandboxError::Timeout { elapsed_ms })
        } else {
            Ok(())
        }
    }

    /// Return the current used-memory count (bytes).
    pub fn used_memory_bytes(&self) -> usize {
        self.used_memory.load(Ordering::Relaxed)
    }

    /// Return the accumulated simulated CPU time (nanoseconds).
    pub fn used_cpu_ns(&self) -> u64 {
        self.used_cpu_ns.load(Ordering::Relaxed)
    }

    /// Capture a point-in-time resource usage snapshot.
    pub fn resource_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot {
            memory_bytes: self.used_memory_bytes(),
            cpu_ns: self.used_cpu_ns(),
            elapsed_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }
}

// ── PluginSandbox ─────────────────────────────────────────────────────────────

/// Wraps plugin execution with policy enforcement.
///
/// `PluginSandbox` owns a `SandboxContext` and exposes helpers that callers
/// should invoke before performing resource-consuming operations within the
/// plugin's execution boundary.
pub struct PluginSandbox {
    ctx: SandboxContext,
}

impl PluginSandbox {
    /// Create a new sandbox from a `SandboxConfig`.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            ctx: SandboxContext::new(config),
        }
    }

    /// Obtain a reference to the inner context.
    pub fn context(&self) -> &SandboxContext {
        &self.ctx
    }

    /// Run `f` inside the sandbox, checking the timeout on entry.
    ///
    /// # Errors
    /// Returns [`SandboxError::Timeout`] if the wall-clock limit is exceeded before
    /// `f` is even called.  Errors from `f` are propagated unchanged.
    pub fn run<F, T>(&self, f: F) -> Result<T, SandboxError>
    where
        F: FnOnce(&SandboxContext) -> Result<T, SandboxError>,
    {
        self.ctx.check_timeout()?;
        f(&self.ctx)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // 1. PermissionSet::new has no permissions
    #[test]
    fn test_perm_set_empty() {
        let p = PermissionSet::new();
        assert!(!p.has(PERM_NETWORK));
        assert!(!p.has(PERM_FILESYSTEM));
    }

    // 2. with_all has all permissions
    #[test]
    fn test_perm_set_all() {
        let p = PermissionSet::with_all();
        assert!(p.has(PERM_NETWORK));
        assert!(p.has(PERM_FILESYSTEM));
        assert!(p.has(PERM_GPU));
        assert!(p.has(PERM_AUDIO));
        assert!(p.has(PERM_VIDEO));
        assert!(p.has(PERM_MEMORY_LARGE));
    }

    // 3. grant adds a permission
    #[test]
    fn test_perm_grant() {
        let p = PermissionSet::new().grant(PERM_NETWORK);
        assert!(p.has(PERM_NETWORK));
        assert!(!p.has(PERM_FILESYSTEM));
    }

    // 4. revoke removes a permission
    #[test]
    fn test_perm_revoke() {
        let p = PermissionSet::with_all().revoke(PERM_NETWORK);
        assert!(!p.has(PERM_NETWORK));
        assert!(p.has(PERM_FILESYSTEM));
    }

    // 5. check_permission success
    #[test]
    fn test_check_permission_ok() {
        let cfg = SandboxConfig {
            permissions: PermissionSet::new().grant(PERM_FILESYSTEM),
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        assert!(ctx.check_permission(PERM_FILESYSTEM).is_ok());
    }

    // 6. check_permission denied
    #[test]
    fn test_check_permission_denied() {
        let ctx = SandboxContext::new(SandboxConfig::default());
        match ctx.check_permission(PERM_NETWORK) {
            Err(SandboxError::PermissionDenied { requested, .. }) => {
                assert_eq!(requested, PERM_NETWORK);
            }
            other => panic!("expected PermissionDenied, got {other:?}"),
        }
    }

    // 7. check_memory within limit
    #[test]
    fn test_check_memory_ok() {
        let cfg = SandboxConfig {
            max_memory_mb: 1,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        assert!(ctx.check_memory(512 * 1024).is_ok()); // 512 KiB < 1 MiB
    }

    // 8. check_memory over limit
    #[test]
    fn test_check_memory_exceeded() {
        let cfg = SandboxConfig {
            max_memory_mb: 1,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        match ctx.check_memory(2 * 1024 * 1024) {
            Err(SandboxError::MemoryExceeded { limit, .. }) => {
                assert_eq!(limit, 1024 * 1024);
            }
            other => panic!("expected MemoryExceeded, got {other:?}"),
        }
    }

    // 9. used_memory accumulates
    #[test]
    fn test_used_memory_accumulates() {
        let cfg = SandboxConfig {
            max_memory_mb: 10,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        ctx.check_memory(1024).expect("first");
        ctx.check_memory(2048).expect("second");
        assert_eq!(ctx.used_memory_bytes(), 3072);
    }

    // 10. release_memory decrements
    #[test]
    fn test_release_memory() {
        let cfg = SandboxConfig {
            max_memory_mb: 10,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        ctx.check_memory(4096).expect("alloc");
        ctx.release_memory(2048);
        assert_eq!(ctx.used_memory_bytes(), 2048);
    }

    // 11. check_timeout within limit
    #[test]
    fn test_check_timeout_ok() {
        let cfg = SandboxConfig {
            timeout_ms: 60_000,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        assert!(ctx.check_timeout().is_ok());
    }

    // 12. check_timeout exceeded (simulate with zero timeout)
    #[test]
    fn test_check_timeout_exceeded() {
        let cfg = SandboxConfig {
            timeout_ms: 0,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        // Sleep 1 ms to ensure elapsed > 0.
        std::thread::sleep(Duration::from_millis(1));
        match ctx.check_timeout() {
            Err(SandboxError::Timeout { elapsed_ms }) => {
                assert!(elapsed_ms >= 1);
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    // 13. PluginSandbox::run propagates fn result
    #[test]
    fn test_plugin_sandbox_run_ok() {
        let sb = PluginSandbox::new(SandboxConfig::permissive());
        let result = sb.run(|ctx| {
            ctx.check_permission(PERM_NETWORK)?;
            Ok(42u32)
        });
        assert_eq!(result.expect("run"), 42);
    }

    // 14. PluginSandbox::run denied permission propagates error
    #[test]
    fn test_plugin_sandbox_run_permission_denied() {
        let sb = PluginSandbox::new(SandboxConfig::default()); // no perms
        let result = sb.run(|ctx| ctx.check_permission(PERM_GPU).map(|_| ()));
        assert!(matches!(result, Err(SandboxError::PermissionDenied { .. })));
    }

    // 15. SandboxError display messages
    #[test]
    fn test_sandbox_error_display() {
        let e = SandboxError::PermissionDenied {
            requested: 0x01,
            available: 0x00,
        };
        assert!(e.to_string().contains("permission denied"));

        let e2 = SandboxError::MemoryExceeded {
            used: 100,
            limit: 50,
        };
        assert!(e2.to_string().contains("memory"));

        let e3 = SandboxError::Timeout { elapsed_ms: 6000 };
        assert!(e3.to_string().contains("timeout"));

        let e4 = SandboxError::CpuExceeded;
        assert!(e4.to_string().contains("CPU"));

        let e5 = SandboxError::PathDenied {
            path: "/etc/passwd".to_string(),
        };
        assert!(e5.to_string().contains("/etc/passwd"));
    }

    // 16. PermissionSet default equals new()
    #[test]
    fn test_perm_set_default() {
        assert_eq!(PermissionSet::default(), PermissionSet::new());
    }

    // 17. Multiple grant/revoke chain
    #[test]
    fn test_perm_chain() {
        let p = PermissionSet::new()
            .grant(PERM_NETWORK)
            .grant(PERM_FILESYSTEM)
            .revoke(PERM_NETWORK);
        assert!(!p.has(PERM_NETWORK));
        assert!(p.has(PERM_FILESYSTEM));
    }

    // 18. permissive config allows all permissions
    #[test]
    fn test_permissive_config() {
        let ctx = SandboxContext::new(SandboxConfig::permissive());
        assert!(ctx.check_permission(PERM_NETWORK).is_ok());
        assert!(ctx.check_permission(PERM_GPU).is_ok());
        assert!(ctx.check_permission(PERM_MEMORY_LARGE).is_ok());
    }

    // 19. memory rollback on failure keeps count unchanged
    #[test]
    fn test_memory_rollback() {
        let cfg = SandboxConfig {
            max_memory_mb: 1,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        ctx.check_memory(512 * 1024).expect("first 512 KiB");
        // This should fail (would exceed 1 MiB).
        let _ = ctx.check_memory(768 * 1024);
        // Used memory should still be only 512 KiB.
        assert_eq!(ctx.used_memory_bytes(), 512 * 1024);
    }

    // 20. SandboxConfig default values
    #[test]
    fn test_default_config() {
        let cfg = SandboxConfig::default();
        assert_eq!(cfg.max_memory_mb, 256);
        assert_eq!(cfg.max_cpu_percent, 50);
        assert_eq!(cfg.timeout_ms, 5_000);
        assert!(!cfg.permissions.has(PERM_NETWORK));
    }

    // 21. Path allow-list: no paths = all allowed (when PERM_FILESYSTEM set)
    #[test]
    fn test_path_empty_allowlist_permits_all() {
        let perms = PermissionSet::new().grant(PERM_FILESYSTEM);
        assert!(perms.is_path_allowed(Path::new("/any/path")));
        assert!(perms.is_path_allowed(Path::new("/etc/hosts")));
    }

    // 22. Path allow-list: restricts access to listed prefix
    #[test]
    fn test_path_allowlist_restricts() {
        let perms = PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .allow_path("/tmp/plugin-data");
        assert!(perms.is_path_allowed(Path::new("/tmp/plugin-data/file.bin")));
        assert!(perms.is_path_allowed(Path::new("/tmp/plugin-data")));
        assert!(!perms.is_path_allowed(Path::new("/etc/passwd")));
        assert!(!perms.is_path_allowed(Path::new("/tmp/other")));
    }

    // 23. No PERM_FILESYSTEM → is_path_allowed returns false
    #[test]
    fn test_path_no_filesystem_perm() {
        let perms = PermissionSet::new().allow_path("/tmp");
        assert!(!perms.is_path_allowed(Path::new("/tmp/file")));
    }

    // 24. check_path success
    #[test]
    fn test_check_path_ok() {
        let cfg = SandboxConfig {
            permissions: PermissionSet::new()
                .grant(PERM_FILESYSTEM)
                .allow_path("/tmp/plugin"),
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        assert!(ctx.check_path(Path::new("/tmp/plugin/data.bin")).is_ok());
    }

    // 25. check_path denied (path not in allow-list)
    #[test]
    fn test_check_path_denied() {
        let cfg = SandboxConfig {
            permissions: PermissionSet::new()
                .grant(PERM_FILESYSTEM)
                .allow_path("/tmp/plugin"),
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        match ctx.check_path(Path::new("/etc/shadow")) {
            Err(SandboxError::PathDenied { path }) => {
                assert!(path.contains("/etc/shadow"));
            }
            other => panic!("expected PathDenied, got {other:?}"),
        }
    }

    // 26. check_path denied (no PERM_FILESYSTEM)
    #[test]
    fn test_check_path_no_fs_perm() {
        let ctx = SandboxContext::new(SandboxConfig::default());
        assert!(matches!(
            ctx.check_path(Path::new("/tmp/any")),
            Err(SandboxError::PermissionDenied { .. })
        ));
    }

    // 27. allow_path / deny_path builder
    #[test]
    fn test_allow_deny_path_builder() {
        let perms = PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .allow_path("/tmp/a")
            .allow_path("/tmp/b")
            .deny_path(Path::new("/tmp/a"));
        assert!(!perms.is_path_allowed(Path::new("/tmp/a/file")));
        assert!(perms.is_path_allowed(Path::new("/tmp/b/file")));
    }

    // 28. charge_cpu_ns: unlimited (max_cpu_ns = 0)
    #[test]
    fn test_cpu_charge_unlimited() {
        let ctx = SandboxContext::new(SandboxConfig::default()); // max_cpu_ns = 0
        ctx.charge_cpu_ns(1_000_000).expect("unlimited");
        ctx.charge_cpu_ns(9_999_999_999).expect("still unlimited");
        assert_eq!(ctx.used_cpu_ns(), 10_000_999_999);
    }

    // 29. charge_cpu_ns: limited, within budget
    #[test]
    fn test_cpu_charge_within_budget() {
        let mut cfg = SandboxConfig::default();
        cfg.max_cpu_ns = 1_000_000;
        let ctx = SandboxContext::new(cfg);
        ctx.charge_cpu_ns(500_000).expect("within");
        ctx.charge_cpu_ns(400_000).expect("still within");
        assert_eq!(ctx.used_cpu_ns(), 900_000);
    }

    // 30. charge_cpu_ns: exceeded rolls back
    #[test]
    fn test_cpu_charge_exceeded() {
        let mut cfg = SandboxConfig::default();
        cfg.max_cpu_ns = 1_000;
        let ctx = SandboxContext::new(cfg);
        ctx.charge_cpu_ns(600).expect("first");
        let err = ctx.charge_cpu_ns(500); // 600+500 > 1000
        assert!(matches!(err, Err(SandboxError::CpuExceeded)));
        // Rolled back — still 600
        assert_eq!(ctx.used_cpu_ns(), 600);
    }

    // 31. resource_snapshot captures all fields
    #[test]
    fn test_resource_snapshot() {
        let cfg = SandboxConfig {
            max_memory_mb: 10,
            max_cpu_ns: 0,
            ..SandboxConfig::default()
        };
        let ctx = SandboxContext::new(cfg);
        ctx.check_memory(1024).expect("mem");
        ctx.charge_cpu_ns(500_000).expect("cpu");
        let snap = ctx.resource_snapshot();
        assert_eq!(snap.memory_bytes, 1024);
        assert_eq!(snap.cpu_ns, 500_000);
        // elapsed_ms should be very small (< 1000 ms)
        assert!(snap.elapsed_ms < 1000);
    }
}
