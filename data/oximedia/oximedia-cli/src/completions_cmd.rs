//! Shell completion script generator for the OxiMedia CLI.

use anyhow::Result;
use clap::CommandFactory;
use clap_complete::{generate, Shell};
use std::io;

use crate::Cli;

/// Generate shell completion scripts and print to stdout.
pub(crate) fn run(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "oximedia", &mut io::stdout());
    Ok(())
}
