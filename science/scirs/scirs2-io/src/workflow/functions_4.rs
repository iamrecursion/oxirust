//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{TaskStatus, TaskType, Workflow, WorkflowState};

/// Workflow visualization
pub mod visualization {
    use super::*;
    /// Workflow visualizer
    pub struct WorkflowVisualizer;
    impl WorkflowVisualizer {
        /// Generate DOT graph representation
        pub fn to_dot(workflow: &Workflow) -> String {
            let mut dot = String::new();
            dot.push_str("digraph workflow {\n");
            dot.push_str("  rankdir=TB;\n");
            dot.push_str("  node [shape=box, style=rounded];\n\n");
            for task in &workflow.tasks {
                let color = match task.task_type {
                    TaskType::DataIngestion => "lightblue",
                    TaskType::Transform => "lightgreen",
                    TaskType::Validation => "yellow",
                    TaskType::MLTraining => "orange",
                    TaskType::MLInference => "pink",
                    TaskType::Export => "lightgray",
                    _ => "white",
                };
                dot.push_str(&format!(
                    "  {} [label=\"{}\", fillcolor={}, style=filled];\n",
                    task.id, task.name, color
                ));
            }
            dot.push('\n');
            for (task_id, deps) in &workflow.dependencies {
                for dep in deps {
                    dot.push_str(&format!("  {dep} -> {task_id};\n"));
                }
            }
            dot.push_str("}\n");
            dot
        }
        /// Generate Mermaid diagram
        pub fn to_mermaid(workflow: &Workflow) -> String {
            let mut mermaid = String::new();
            mermaid.push_str("graph TD\n");
            for task in &workflow.tasks {
                let shape = match task.task_type {
                    TaskType::DataIngestion => "[",
                    TaskType::Transform => "(",
                    TaskType::Validation => "{",
                    TaskType::MLTraining => "[[",
                    TaskType::MLInference => "((",
                    TaskType::Export => "[",
                    _ => "[",
                };
                let close = match task.task_type {
                    TaskType::DataIngestion => "]",
                    TaskType::Transform => ")",
                    TaskType::Validation => "}",
                    TaskType::MLTraining => "]]",
                    TaskType::MLInference => "))",
                    TaskType::Export => "]",
                    _ => "]",
                };
                mermaid.push_str(&format!("    {}{}{}{}\n", task.id, shape, task.name, close));
            }
            for (task_id, deps) in &workflow.dependencies {
                for dep in deps {
                    mermaid.push_str(&format!("    {dep} --> {task_id}\n"));
                }
            }
            mermaid
        }
        /// Generate execution timeline
        pub fn execution_timeline(state: &WorkflowState) -> String {
            let mut timeline = String::new();
            timeline.push_str("gantt\n");
            timeline.push_str("    title Workflow Execution Timeline\n");
            timeline.push_str("    dateFormat YYYY-MM-DD HH:mm:ss\n\n");
            let mut tasks: Vec<_> = state.task_states.iter().collect();
            tasks.sort_by_key(|(_, state)| state.start_time);
            for (task_id, task_state) in tasks {
                if let (Some(start), Some(end)) = (task_state.start_time, task_state.end_time) {
                    let status = match task_state.status {
                        TaskStatus::Success => "done",
                        TaskStatus::Failed => "crit",
                        TaskStatus::Running => "active",
                        _ => "",
                    };
                    timeline.push_str(&format!(
                        "    {} :{}, {}, {}\n",
                        task_id,
                        status,
                        start.format("%Y-%m-%d %H:%M:%S"),
                        end.format("%Y-%m-%d %H:%M:%S")
                    ));
                }
            }
            timeline
        }
    }
}
