use super::*;

use super::*;

#[test]
fn testworkflow_builder() {
    let workflow = WorkflowBuilder::new("test_wf", "Test Workflow")
        .description("A test workflow")
        .add_task(
            tasks::data_ingestion("task1", "Load Data")
                .output("data.csv")
                .build(),
        )
        .add_task(
            tasks::transform("task2", "Process Data")
                .input("data.csv")
                .output("processed.csv")
                .build(),
        )
        .add_dependency("task2", "task1")
        .build()
        .expect("Operation failed");

    assert_eq!(workflow.tasks.len(), 2);
    assert_eq!(
        workflow
            .dependencies
            .get("task2")
            .expect("Operation failed"),
        &vec!["task1".to_string()]
    );
}

#[test]
fn test_cycle_detection() {
    let result = WorkflowBuilder::new("cyclic", "Cyclic Workflow")
            .add_task(tasks::transform("a", "Task A").build())
            .add_task(tasks::transform("b", "Task B").build())
            .add_task(tasks::transform("c", "Task C").build())
            .add_dependency("a", "b")
            .add_dependency("b", "c")
            .add_dependency("c", "a") // Creates cycle
            .build();

    assert!(result.is_err());
}

#[test]
fn test_etl_template() {
    let workflow = templates::etlworkflow("My ETL Pipeline")
        .build()
        .expect("Operation failed");

    assert_eq!(workflow.tasks.len(), 4);
    assert!(workflow.tasks.iter().any(|t| t.id == "extract"));
    assert!(workflow.tasks.iter().any(|t| t.id == "transform"));
    assert!(workflow.tasks.iter().any(|t| t.id == "validate"));
    assert!(workflow.tasks.iter().any(|t| t.id == "load"));
}
