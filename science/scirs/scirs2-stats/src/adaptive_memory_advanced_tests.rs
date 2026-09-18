use super::*;
use std::time::Duration;

#[test]
fn test_adaptive_memory_manager_creation() {
    let manager = AdaptiveMemoryManager::<f64>::new();
    let stats = manager.get_memory_stats();
    assert_eq!(stats.total_allocated, 0);
}

#[test]
fn test_memory_allocation() {
    let manager = AdaptiveMemoryManager::<f64>::new();
    let ptr = manager.allocate(1024).expect("Test: operation failed");
    assert!(!ptr.is_null());

    let result = manager.deallocate(ptr, 1024);
    assert!(result.is_ok());
}

#[test]
fn test_performance_metrics() {
    let manager = AdaptiveMemoryManager::<f64>::new();
    let metrics = manager.get_performance_metrics();

    assert!(metrics.allocation_rate > 0.0);
    assert!(metrics.cache_hit_ratio >= 0.0 && metrics.cache_hit_ratio <= 1.0);
    assert!(metrics.numa_locality >= 0.0 && metrics.numa_locality <= 1.0);
}

#[test]
fn test_gc_trigger() {
    let manager = AdaptiveMemoryManager::<f64>::new();
    let result = manager.trigger_gc().expect("Test: operation failed");

    assert!(result.memory_reclaimed > 0);
    assert!(result.collection_time > Duration::from_nanos(0));
}

#[test]
fn test_config_update() {
    let mut manager = AdaptiveMemoryManager::<f64>::new();
    let mut new_config = AdaptiveMemoryConfig::default();
    new_config.allocation_strategy = AllocationStrategy::NumaAware;

    manager.update_config(new_config);
    assert!(matches!(
        manager.get_config().allocation_strategy,
        AllocationStrategy::NumaAware
    ));
}
