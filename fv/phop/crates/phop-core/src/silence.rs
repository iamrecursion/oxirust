//! Scoped suppression of process `stdout` at the file-descriptor level.
//!
//! Some dependency operations in `scirs2-autograd` (notably the gradient-seed path through
//! `ConvertToTensor`) emit unconditional `println!` debug output. phop has no control over
//! that, so during the tight optimization loop we redirect fd 1 to `/dev/null` and restore it
//! afterwards. This keeps phop's own output clean.
//!
//! Scope it as narrowly as possible: it is process-global while alive, so only wrap
//! compute-only regions that intentionally produce no stdout of their own. (Under
//! `cargo nextest`, each test runs in its own process, so this does not cross test
//! boundaries.)

/// RAII guard that silences `stdout` for its lifetime, restoring it on drop.
#[cfg(unix)]
pub struct SilenceStdout {
    saved_fd: i32,
}

#[cfg(unix)]
impl SilenceStdout {
    /// Begin suppressing stdout. Returns `None` if the OS handles could not be acquired,
    /// in which case output is simply not suppressed (never fatal).
    #[must_use]
    pub fn new() -> Option<Self> {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        // SAFETY: standard dup/open/dup2 dance on the stdout file descriptor; every raw fd
        // obtained is checked and released, and the original is restored on drop.
        unsafe {
            let saved_fd = libc::dup(libc::STDOUT_FILENO);
            if saved_fd < 0 {
                return None;
            }
            let devnull = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if devnull < 0 {
                libc::close(saved_fd);
                return None;
            }
            if libc::dup2(devnull, libc::STDOUT_FILENO) < 0 {
                libc::close(devnull);
                libc::close(saved_fd);
                return None;
            }
            libc::close(devnull);
            Some(Self { saved_fd })
        }
    }
}

#[cfg(unix)]
impl Drop for SilenceStdout {
    fn drop(&mut self) {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        // SAFETY: `saved_fd` is a valid descriptor we created in `new`; restore and close it.
        unsafe {
            libc::dup2(self.saved_fd, libc::STDOUT_FILENO);
            libc::close(self.saved_fd);
        }
    }
}

/// No-op silencer on non-unix platforms (e.g. `wasm32`), where there is no fd-level stdout to
/// redirect.
#[cfg(not(unix))]
pub struct SilenceStdout;

#[cfg(not(unix))]
impl SilenceStdout {
    /// Always returns `Some` (no-op).
    #[must_use]
    pub fn new() -> Option<Self> {
        Some(Self)
    }
}

// A trivial `Drop` so that `Option<SilenceStdout>` carries drop glue on every platform; without
// it, an explicit `drop(_silencer)` on the no-op variant trips `clippy::drop_non_drop` under the
// `wasm32` build.
#[cfg(not(unix))]
impl Drop for SilenceStdout {
    fn drop(&mut self) {}
}
