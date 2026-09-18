//! Unix man page generator for the OxiMedia CLI.

use anyhow::Result;
use clap::CommandFactory;
use std::io::Write;

use crate::Cli;

/// Render a Unix man page to stdout.
pub(crate) fn run() -> Result<()> {
    let cmd = Cli::command();
    let man = clap_mangen::Man::new(cmd);
    let mut buf = Vec::new();
    man.render(&mut buf)?;
    std::io::stdout().write_all(&buf)?;
    Ok(())
}
