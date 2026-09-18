//! Shared helpers for `oxixml-cli`'s integration tests.
//!
//! Every test in this suite drives the crate through its public library entry point,
//! [`oxixml_cli::run_status`] — exactly what `main` does, minus spawning a process — so what is
//! under test here is the same code path a real invocation of the `oxixml` binary runs.
//!
//! Each `tests/*.rs` file is compiled as its own independent binary with its own copy of this
//! module, and no single one of them exercises every helper here — hence the blanket
//! `dead_code` allow, standard practice for a shared test-support module.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Runs `oxixml <args>` with `stdin` as its standard input, returning
/// `(exit_status, stdout, stderr)` with stdout as raw bytes — the one output this suite's
/// `hdt build`/`hdt dump` tests need, since HDT is a binary format.
pub fn run_bytes(args: &[&str], stdin: &[u8]) -> (u8, Vec<u8>, String) {
    let owned_args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    let mut stdin_reader = stdin;
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = oxixml_cli::run_status(&owned_args, &mut stdin_reader, &mut stdout, &mut stderr);
    (
        status,
        stdout,
        String::from_utf8(stderr).expect("stderr is utf8"),
    )
}

/// [`run_bytes`], decoding stdout as UTF-8 text — every subcommand's output in this suite is
/// text except HDT's, which uses [`run_bytes`] directly.
///
/// # Panics
///
/// If stdout is not valid UTF-8, which for every caller of this particular helper signals a
/// real bug rather than an expected outcome.
pub fn run(args: &[&str], stdin: &[u8]) -> (u8, String, String) {
    let (status, stdout, stderr) = run_bytes(args, stdin);
    (
        status,
        String::from_utf8(stdout).expect("stdout is utf8"),
        stderr,
    )
}

/// [`run`] with empty stdin — the common case for a subcommand whose input is a named file, not
/// `-`.
pub fn run_no_stdin(args: &[&str]) -> (u8, String, String) {
    run(args, &[])
}

/// A uniquely-named file under [`std::env::temp_dir`] that removes itself when dropped, so a
/// test can write a fixture without leaking it into the system temp directory whether the test
/// passes, fails, or panics.
pub struct TempFile {
    path: PathBuf,
}

impl TempFile {
    /// Writes `contents` to a fresh file named `<label>-<unique>.<extension>` under
    /// [`std::env::temp_dir`]. `label` is typically the calling test's own name, to make a
    /// leaked file (should cleanup ever be skipped, e.g. by a hard process kill) traceable back
    /// to its test.
    #[must_use]
    pub fn new(label: &str, extension: &str, contents: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "oxixml-cli-it-{label}-{}-{}.{extension}",
            std::process::id(),
            next_unique_id(),
        ));
        std::fs::write(&path, contents).expect("write fixture file");
        Self { path }
    }

    /// The file's path, as `str` (every path this constructs is built from UTF-8 parts, so this
    /// never fails in practice).
    #[must_use]
    pub fn path(&self) -> &str {
        self.path.to_str().expect("fixture path is utf8")
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A process-wide counter, so two [`TempFile`]s created in the same millisecond by concurrently
/// running tests (`cargo test` runs tests in parallel by default) never collide.
fn next_unique_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}
