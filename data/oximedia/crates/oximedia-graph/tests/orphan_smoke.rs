//! Smoke tests for newly-wired orphan modules in oximedia-graph.

#[test]
fn test_async_exec_config_default() {
    use oximedia_graph::async_exec::AsyncExecutorConfig;
    let config = AsyncExecutorConfig::default();
    assert!(config.stage_timeout.is_some());
}

#[test]
fn test_graph_evaluator_accessible() {
    let _ = std::any::type_name::<oximedia_graph::graph_evaluator::GraphEvaluator>();
}

#[test]
fn test_graph_rewrite_engine() {
    use oximedia_graph::graph_rewrite::RewriteEngine;
    let engine = RewriteEngine::new();
    assert_eq!(engine.rule_count(), 0);
}

#[test]
fn test_node_priority_manager() {
    use oximedia_graph::node_priority::PriorityManager;
    let pm = PriorityManager::new();
    assert_eq!(pm.count(), 0);
}

#[test]
fn test_port_buffer_strategy() {
    use oximedia_graph::port_buffer::{BufferStrategy, PortBuffer};
    let buf = PortBuffer::new(BufferStrategy::Unbounded, "test");
    assert_eq!(buf.len(), 0);
}
