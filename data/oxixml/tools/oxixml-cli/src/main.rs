//! `oxixml` — thin process entry point.
//!
//! Everything this binary does lives in the `oxixml-cli` library crate, whose [`run`](oxixml_cli::run)
//! is testable directly without spawning a process; this file only wires it to the real
//! `std::env`/`std::io` handles.

use std::io;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let stderr = io::stderr();
    oxixml_cli::run(
        args,
        &mut stdin.lock(),
        &mut stdout.lock(),
        &mut stderr.lock(),
    )
}
