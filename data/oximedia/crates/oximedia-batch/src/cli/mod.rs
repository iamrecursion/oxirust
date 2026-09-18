//! Command-line interface for batch processing

use crate::error::Result;
use crate::job::{BatchJob, BatchOperation, InputSpec, OutputSpec};
use crate::operations::{FileOperation, OutputFormat};
use crate::{BatchEngine, JobId};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

/// CLI arguments
#[derive(Parser, Debug)]
#[command(name = "oximedia-batch")]
#[command(about = "Batch processing engine for OxiMedia", long_about = None)]
pub struct CliArgs {
    /// Database path
    #[arg(long, default_value = "batch.db")]
    pub db_path: String,

    /// Number of worker threads
    #[arg(long, default_value = "4")]
    pub workers: usize,

    /// Subcommand
    #[command(subcommand)]
    pub command: Commands,
}

/// CLI commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Submit a new job
    Submit {
        /// Job name
        #[arg(long)]
        name: String,

        /// Operation type
        #[arg(long)]
        operation: String,

        /// Input file pattern
        #[arg(long)]
        input: String,

        /// Output file pattern
        #[arg(long)]
        output: Option<String>,

        /// Template name
        #[arg(long)]
        template: Option<String>,
    },

    /// List all jobs
    List {
        /// Filter by status
        #[arg(long)]
        status: Option<String>,
    },

    /// Get job status
    Status {
        /// Job ID
        job_id: String,
    },

    /// Cancel a job
    Cancel {
        /// Job ID
        job_id: String,
    },

    /// Watch a folder for new files
    Watch {
        /// Folder to watch
        #[arg(long)]
        folder: PathBuf,

        /// Template to use
        #[arg(long)]
        template: String,
    },

    /// Start the API server
    Serve {
        /// Server address
        #[arg(long, default_value = "0.0.0.0:3000")]
        addr: String,
    },
}

/// Execute CLI command
///
/// # Arguments
///
/// * `args` - CLI arguments
///
/// # Errors
///
/// Returns an error if command execution fails
pub async fn execute(args: CliArgs) -> Result<()> {
    let engine = Arc::new(BatchEngine::new(&args.db_path, args.workers)?);
    engine.start().await?;

    match args.command {
        Commands::Submit {
            name,
            operation,
            input,
            output,
            template: _,
        } => {
            let op = parse_operation(&operation)?;
            let mut job = BatchJob::new(name, op);

            job.add_input(InputSpec::new(input));

            if let Some(output_path) = output {
                job.add_output(OutputSpec::new(output_path, OutputFormat::Mp4));
            }

            let job_id = engine.submit_job(job).await?;
            println!("Job submitted: {job_id}");
        }

        Commands::List { status: _ } => {
            let jobs = engine.list_jobs()?;
            for job in jobs {
                println!("{} - {}", job.id, job.name);
            }
        }

        Commands::Status { job_id } => {
            let job_id = JobId::from_string(job_id);
            let status = engine.get_job_status(&job_id).await?;
            println!("Status: {status}");
        }

        Commands::Cancel { job_id } => {
            let job_id = JobId::from_string(job_id);
            engine.cancel_job(&job_id).await?;
            println!("Job cancelled");
        }

        Commands::Watch { folder, template } => {
            println!("Watching folder: {}", folder.display());
            let watch_config = crate::watch::WatchConfig::new(folder)
                .with_template(template)
                .with_pattern("*.mp4".to_string());
            let watch_folder = crate::watch::WatchFolder::new(watch_config, engine);
            watch_folder.start().await?;
        }

        Commands::Serve { addr } => {
            println!("Starting API server on {addr}");
            crate::api::start_server(engine, &addr).await?;
        }
    }

    Ok(())
}

fn parse_operation(operation: &str) -> Result<BatchOperation> {
    match operation {
        "copy" => Ok(BatchOperation::FileOp {
            operation: FileOperation::Copy { overwrite: false },
        }),
        "move" => Ok(BatchOperation::FileOp {
            operation: FileOperation::Move { overwrite: false },
        }),
        "transcode" => Ok(BatchOperation::Transcode {
            preset: "default".to_string(),
        }),
        "qc" => Ok(BatchOperation::QualityCheck {
            profile: "default".to_string(),
        }),
        _ => Err(crate::error::BatchError::ValidationError(format!(
            "Unknown operation: {operation}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_operation_copy() {
        let result = parse_operation("copy");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_operation_move() {
        let result = parse_operation("move");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_operation_transcode() {
        let result = parse_operation("transcode");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_operation_qc() {
        let result = parse_operation("qc");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_operation_unknown() {
        let result = parse_operation("unknown");
        assert!(result.is_err());
    }
}
