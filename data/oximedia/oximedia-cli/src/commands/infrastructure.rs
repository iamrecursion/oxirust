//! Infrastructure-domain sub-enum definitions used by `Commands`.
//!
//! Currently houses [`MonitorCommand`]. The other infrastructure-domain
//! variants on the top-level `Commands` enum (`Distributed`, `Farm`,
//! `Renderfarm`, `Workflow`, `Collab`, `Proxy`, `Profiler`, `Recommend`,
//! `Auto`, `Plugin`, `Stream`, `Gaming`, `Ml`) delegate to sub-enums that
//! live alongside their handler modules and are therefore not re-defined
//! here.

use clap::Subcommand;
use std::path::PathBuf;

/// Monitor subcommands.
#[derive(Subcommand)]
pub(crate) enum MonitorCommand {
    /// Start monitoring a stream or file
    Start {
        /// Target to monitor (file path or stream URL)
        #[arg(value_name = "TARGET")]
        target: String,

        /// Database path for metric storage
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Metrics collection interval in milliseconds
        #[arg(long, default_value = "1000")]
        interval_ms: u64,

        /// Enable system metrics (CPU, memory, disk)
        #[arg(long)]
        system_metrics: bool,

        /// Enable quality metrics (PSNR, SSIM, bitrate)
        #[arg(long)]
        quality_metrics: bool,
    },

    /// Show current monitoring status
    Status {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Show detailed per-component status
        #[arg(long)]
        detailed: bool,
    },

    /// Show recent alerts
    Alerts {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Number of recent alerts to show
        #[arg(long, default_value = "20")]
        count: usize,

        /// Filter by severity: info, warning, error, critical
        #[arg(long)]
        severity: Option<String>,
    },

    /// Configure monitoring thresholds
    Config {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// CPU alert threshold (percentage)
        #[arg(long)]
        cpu_threshold: Option<f64>,

        /// Memory alert threshold (percentage)
        #[arg(long)]
        memory_threshold: Option<f64>,

        /// Quality score alert threshold (0-100)
        #[arg(long)]
        quality_threshold: Option<f64>,

        /// Show current configuration only
        #[arg(long)]
        show: bool,
    },

    /// Display monitoring dashboard
    Dashboard {
        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Refresh interval in seconds
        #[arg(long, default_value = "5")]
        refresh_secs: u64,

        /// Number of history points to display
        #[arg(long, default_value = "60")]
        history_points: usize,
    },
}
