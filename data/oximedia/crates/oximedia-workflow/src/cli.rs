//! Command-line interface for workflow management.

use crate::error::Result;
use crate::executor::{DefaultTaskExecutor, WorkflowExecutor};
use crate::persistence::PersistenceManager;
use crate::scheduler::{Trigger, WorkflowScheduler};
use crate::workflow::{Workflow, WorkflowId};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// Workflow management CLI.
#[derive(Debug, Parser)]
#[command(name = "oximedia-workflow")]
#[command(about = "OxiMedia workflow orchestration tool")]
#[command(version)]
pub struct Cli {
    /// Database path.
    #[arg(short, long, default_value = "workflows.db")]
    pub database: PathBuf,

    /// Verbose output.
    #[arg(short, long)]
    pub verbose: bool,

    /// The command to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// CLI commands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Create a new workflow from YAML.
    Create {
        /// Workflow definition file (YAML).
        #[arg(short, long)]
        file: PathBuf,
    },

    /// List all workflows.
    List,

    /// Show workflow details.
    Show {
        /// Workflow ID.
        id: String,
    },

    /// Execute a workflow.
    Execute {
        /// Workflow ID.
        id: String,

        /// Wait for completion.
        #[arg(short, long)]
        wait: bool,
    },

    /// Delete a workflow.
    Delete {
        /// Workflow ID.
        id: String,
    },

    /// Schedule a workflow.
    Schedule {
        /// Workflow ID.
        id: String,

        /// Cron expression.
        #[arg(short, long)]
        cron: Option<String>,

        /// Interval in seconds.
        #[arg(short, long)]
        interval: Option<u64>,
    },

    /// List schedules.
    Schedules,

    /// Export workflow to YAML.
    Export {
        /// Workflow ID.
        id: String,

        /// Output file.
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Import workflow from YAML.
    Import {
        /// Input file.
        file: PathBuf,
    },

    /// Validate workflow.
    Validate {
        /// Workflow file.
        file: PathBuf,
    },

    /// Start API server.
    Serve {
        /// Server address.
        #[arg(short, long, default_value = "0.0.0.0:3000")]
        addr: String,
    },

    /// Show workflow status.
    Status {
        /// Workflow ID.
        id: String,
    },

    /// Cancel running workflow.
    Cancel {
        /// Workflow ID.
        id: String,
    },
}

impl Cli {
    /// Run the CLI.
    pub async fn run(self) -> Result<()> {
        // Initialize tracing
        if self.verbose {
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::DEBUG)
                .init();
        } else {
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::INFO)
                .init();
        }

        let persistence = Arc::new(PersistenceManager::new(&self.database)?);

        match self.command {
            Commands::Create { file } => {
                let workflow = load_workflow_from_file(&file)?;
                persistence.save_workflow(&workflow)?;
                println!("Created workflow: {}", workflow.id);
                println!("Name: {}", workflow.name);
                println!("Tasks: {}", workflow.tasks.len());
            }

            Commands::List => {
                let workflows = persistence.list_workflows()?;
                if workflows.is_empty() {
                    println!("No workflows found.");
                } else {
                    println!("Workflows:");
                    for id in workflows {
                        if let Ok(workflow) = persistence.load_workflow(id) {
                            println!(
                                "  {} - {} ({} tasks, state: {:?})",
                                id,
                                workflow.name,
                                workflow.tasks.len(),
                                workflow.state
                            );
                        }
                    }
                }
            }

            Commands::Show { id } => {
                let workflow_id = parse_workflow_id(&id)?;
                let workflow = persistence.load_workflow(workflow_id)?;

                println!("Workflow: {}", workflow.id);
                println!("Name: {}", workflow.name);
                println!("Description: {}", workflow.description);
                println!("State: {:?}", workflow.state);
                println!("Tasks: {}", workflow.tasks.len());
                println!("Edges: {}", workflow.edges.len());
                println!("\nTasks:");
                for task in workflow.tasks.values() {
                    println!(
                        "  {} - {} (state: {:?}, priority: {:?})",
                        task.id, task.name, task.state, task.priority
                    );
                }
            }

            Commands::Execute { id, wait: _ } => {
                let workflow_id = parse_workflow_id(&id)?;
                let mut workflow = persistence.load_workflow(workflow_id)?;

                println!("Executing workflow: {}", workflow.name);

                let executor = WorkflowExecutor::new(Arc::new(DefaultTaskExecutor));
                let result = executor.execute(&mut workflow).await?;

                println!("Workflow completed with state: {:?}", result.state);
                println!("Duration: {:?}", result.duration);
                println!("Tasks completed: {}", result.task_results.len());

                persistence.save_workflow(&workflow)?;
            }

            Commands::Delete { id } => {
                let workflow_id = parse_workflow_id(&id)?;
                persistence.delete_workflow(workflow_id)?;
                println!("Deleted workflow: {workflow_id}");
            }

            Commands::Schedule { id, cron, interval } => {
                let workflow_id = parse_workflow_id(&id)?;
                let workflow = persistence.load_workflow(workflow_id)?;

                let trigger = if let Some(cron_expr) = cron {
                    Trigger::Cron {
                        expression: cron_expr,
                        timezone: "UTC".to_string(),
                    }
                } else if let Some(secs) = interval {
                    Trigger::Interval { seconds: secs }
                } else {
                    return Err(crate::error::WorkflowError::InvalidConfiguration(
                        "Either --cron or --interval must be specified".to_string(),
                    ));
                };

                let scheduler = WorkflowScheduler::new();
                scheduler.add_schedule(workflow, trigger).await?;

                println!("Scheduled workflow: {workflow_id}");
            }

            Commands::Schedules => {
                let scheduler = WorkflowScheduler::new();
                let schedules = scheduler.list_schedules().await;

                if schedules.is_empty() {
                    println!("No schedules found.");
                } else {
                    println!("Scheduled workflows:");
                    for (id, schedule) in schedules {
                        println!(
                            "  {} - {} (enabled: {}, next: {:?})",
                            id, schedule.workflow.name, schedule.enabled, schedule.next_execution
                        );
                    }
                }
            }

            Commands::Export { id, output } => {
                let workflow_id = parse_workflow_id(&id)?;
                let workflow = persistence.load_workflow(workflow_id)?;

                let yaml = serde_yaml::to_string(&workflow)?;
                std::fs::write(&output, yaml)?;

                println!("Exported workflow to: {}", output.display());
            }

            Commands::Import { file } => {
                let workflow = load_workflow_from_file(&file)?;
                persistence.save_workflow(&workflow)?;

                println!("Imported workflow: {}", workflow.id);
                println!("Name: {}", workflow.name);
            }

            Commands::Validate { file } => {
                let workflow = load_workflow_from_file(&file)?;
                workflow.validate()?;

                println!("Workflow is valid!");
                println!("Name: {}", workflow.name);
                println!("Tasks: {}", workflow.tasks.len());
                println!("Edges: {}", workflow.edges.len());

                if workflow.has_cycle() {
                    println!("WARNING: Workflow contains a cycle!");
                } else {
                    println!("No cycles detected.");
                }
            }

            Commands::Serve { addr } => {
                use crate::api;

                let scheduler = Arc::new(WorkflowScheduler::new());
                let monitoring = Arc::new(crate::monitoring::MonitoringService::new());
                let executor = Arc::new(DefaultTaskExecutor);

                let state = api::ApiState {
                    persistence: persistence.clone(),
                    scheduler,
                    monitoring,
                    executor,
                    active_workflows: Arc::new(tokio::sync::RwLock::new(
                        std::collections::HashMap::new(),
                    )),
                };

                let app = api::create_router(state);

                info!("Starting API server on {} (Ctrl-C / SIGTERM to stop)", addr);
                let listener = tokio::net::TcpListener::bind(&addr).await?;

                // Serve with graceful shutdown on SIGINT (Ctrl-C) or SIGTERM.
                axum::serve(listener, app)
                    .with_graceful_shutdown(shutdown_signal())
                    .await?;

                info!("API server stopped gracefully");
            }

            Commands::Status { id } => {
                let workflow_id = parse_workflow_id(&id)?;
                let workflow = persistence.load_workflow(workflow_id)?;

                println!("Workflow Status:");
                println!("  ID: {}", workflow.id);
                println!("  Name: {}", workflow.name);
                println!("  State: {:?}", workflow.state);
                println!("  Tasks: {}", workflow.tasks.len());

                let mut pending = 0;
                let mut running = 0;
                let mut completed = 0;
                let mut failed = 0;

                for task in workflow.tasks.values() {
                    match task.state {
                        crate::task::TaskState::Pending | crate::task::TaskState::Queued => {
                            pending += 1;
                        }
                        crate::task::TaskState::Running => running += 1,
                        crate::task::TaskState::Completed => completed += 1,
                        crate::task::TaskState::Failed => failed += 1,
                        _ => {}
                    }
                }

                println!("\nTask Summary:");
                println!("  Pending: {pending}");
                println!("  Running: {running}");
                println!("  Completed: {completed}");
                println!("  Failed: {failed}");
            }

            Commands::Cancel { id } => {
                let workflow_id = parse_workflow_id(&id)?;
                println!("Workflow {workflow_id} cancelled");
                // In a real implementation, this would signal the running workflow
            }
        }

        Ok(())
    }
}

/// Wait for a shutdown signal (SIGINT or SIGTERM) and return.
///
/// This is used with `axum::serve(...).with_graceful_shutdown(shutdown_signal())`
/// to allow the server to drain in-flight requests before exiting.
async fn shutdown_signal() {
    // SIGINT: Ctrl-C from the terminal.
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install SIGINT handler");
    };

    // SIGTERM: sent by process managers (systemd, Docker, Kubernetes).
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    // On non-Unix platforms (Windows) there is no SIGTERM, so we just wait
    // for Ctrl-C.
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            tracing::info!("Received SIGINT (Ctrl-C) – initiating graceful shutdown");
        }
        () = terminate => {
            tracing::info!("Received SIGTERM – initiating graceful shutdown");
        }
    }
}

fn load_workflow_from_file(path: &PathBuf) -> Result<Workflow> {
    let content = std::fs::read_to_string(path)?;
    let workflow: Workflow = serde_yaml::from_str(&content)?;
    Ok(workflow)
}

fn parse_workflow_id(id_str: &str) -> Result<WorkflowId> {
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|_| {
        crate::error::WorkflowError::InvalidParameter {
            param: "workflow_id".to_string(),
            value: id_str.to_string(),
        }
    })?;
    Ok(WorkflowId::from(uuid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workflow_id() {
        let uuid = uuid::Uuid::new_v4();
        let id_str = uuid.to_string();
        let result = parse_workflow_id(&id_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_invalid_workflow_id() {
        let result = parse_workflow_id("invalid");
        assert!(result.is_err());
    }
}
