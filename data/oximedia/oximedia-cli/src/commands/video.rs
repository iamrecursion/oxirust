//! Video-domain sub-enum definitions used by `Commands`.
//!
//! Currently houses [`RestoreCommand`]. The other video-domain variants
//! on the top-level `Commands` enum (`Scene`, `Scopes`, `Quality`,
//! `Denoise`, `Stabilize`, `Lut`, `Color`, `DolbyVision`, `Filter`,
//! `Scaling`) delegate to sub-enums that live alongside their handler
//! modules (e.g. `scene::SceneCommand`, `lut_cmd::LutCommand`) and are
//! therefore not re-defined here.

use clap::Subcommand;
use std::path::PathBuf;

/// Restore subcommands.
#[derive(Subcommand)]
pub(crate) enum RestoreCommand {
    /// Restore degraded audio
    Audio {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Restoration mode: vinyl, tape, broadcast, archival, custom
        #[arg(long, default_value = "vinyl")]
        mode: String,

        /// Sample rate override (Hz)
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Enable declipping
        #[arg(long)]
        declip: bool,

        /// Enable decrackle
        #[arg(long)]
        decrackle: bool,

        /// Enable hum removal
        #[arg(long)]
        dehum: bool,

        /// Enable noise reduction
        #[arg(long)]
        denoise: bool,

        /// Treat input as raw PCM float32 LE (skip format detection)
        #[arg(long, help = "Treat input as raw PCM float32 LE")]
        raw: bool,
    },

    /// Restore degraded video
    Video {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Restoration mode: deinterlace, upscale, stabilize, color-correct, full
        #[arg(long, default_value = "full")]
        mode: String,

        /// Target width
        #[arg(long)]
        width: Option<u32>,

        /// Target height
        #[arg(long)]
        height: Option<u32>,
    },

    /// Analyze degradation type and severity
    Analyze {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Analysis type: audio, video, auto
        #[arg(long, default_value = "auto")]
        analysis_type: String,
    },

    /// Batch restore multiple files
    Batch {
        /// Input directory
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Restoration mode
        #[arg(long, default_value = "vinyl")]
        mode: String,

        /// File extension filter
        #[arg(long)]
        extension: Option<String>,
    },

    /// Compare before/after quality
    Compare {
        /// Original (degraded) file
        #[arg(long)]
        original: PathBuf,

        /// Restored file
        #[arg(long)]
        restored: PathBuf,
    },
}
