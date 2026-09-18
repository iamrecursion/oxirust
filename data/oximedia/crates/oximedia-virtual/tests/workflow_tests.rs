//! Workflow-specific integration tests

use oximedia_virtual::{
    workflows::{WorkflowSession, WorkflowState},
    WorkflowType,
};

#[test]
fn test_workflow_session_lifecycle() {
    let mut session = WorkflowSession::new("test-session".to_string(), WorkflowType::LedWall);

    assert_eq!(session.state, WorkflowState::Idle);
    assert_eq!(session.frame_count, 0);

    session.start(0);
    assert_eq!(session.state, WorkflowState::Recording);

    for _ in 0..100 {
        session.next_frame();
    }

    assert_eq!(session.frame_count, 100);

    session.stop();
    assert_eq!(session.state, WorkflowState::Idle);
}

#[test]
fn test_workflow_session_multiple_starts() {
    let mut session = WorkflowSession::new("multi-start".to_string(), WorkflowType::Hybrid);

    session.start(0);
    session.next_frame();
    session.next_frame();
    assert_eq!(session.frame_count, 2);

    session.stop();
    session.start(1000);
    assert_eq!(session.frame_count, 0); // Reset on restart
}

#[test]
fn test_all_workflow_types() {
    let workflows = [
        WorkflowType::LedWall,
        WorkflowType::Hybrid,
        WorkflowType::GreenScreen,
        WorkflowType::AugmentedReality,
    ];

    for workflow_type in &workflows {
        let session = WorkflowSession::new(format!("{workflow_type:?}"), *workflow_type);
        assert_eq!(session.workflow_type, *workflow_type);
    }
}
