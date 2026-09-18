//! Compatibility/tooling-domain sub-enum definitions used by `Commands`.
//!
//! Currently houses [`PresetCommand`] (preset management subcommands for
//! the `oximedia preset` family). The other compatibility-domain variants
//! on the top-level `Commands` enum (`Ffcompat`, `Tui`, `Completions`,
//! `ManPage`) carry their argument parsers inline and need no separate
//! sub-enum.

use clap::Subcommand;
use std::path::PathBuf;

/// Preset management subcommands.
#[derive(Subcommand)]
pub(crate) enum PresetCommand {
    /// List all available presets
    List {
        /// Filter by category (web, device, quality, archival, streaming, custom)
        #[arg(short, long)]
        category: Option<String>,

        /// Show detailed information
        #[arg(long)]
        detail: bool,
    },

    /// Show detailed information about a preset
    Show {
        /// Preset name
        #[arg(value_name = "NAME")]
        name: String,

        /// Output as TOML
        #[arg(long)]
        toml: bool,
    },

    /// Create a new custom preset interactively
    Create {
        /// Output directory for custom presets
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a preset template file
    Template {
        /// Output file path
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },

    /// Import a preset from a TOML file
    Import {
        /// Input TOML file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Export a preset to a TOML file
    Export {
        /// Preset name
        #[arg(value_name = "NAME")]
        name: String,

        /// Output file path
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },

    /// Remove a custom preset
    Remove {
        /// Preset name
        #[arg(value_name = "NAME")]
        name: String,
    },
}
