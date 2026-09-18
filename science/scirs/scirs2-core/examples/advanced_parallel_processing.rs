//! Example demonstrating advanced parallel processing features in scirs2-core
//!
//! This example shows:
//! - Custom partitioning strategies for different data distributions
//! - Work-stealing scheduler usage
//! - Nested parallelism with resource control
//! - Load balancing and adaptive scheduling

use scirs2_core::error::CoreResult;
#[cfg(feature = "parallel")]
use scirs2_core::parallel::{
    nested_scope, nested_scope_with_limits, with_nested_policy, DataPartitioner, LoadBalancer,
    NestedConfig, NestedPolicy, PartitionerConfig, ResourceLimits, SchedulerConfigBuilder,
    WorkStealingScheduler,
};
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() -> CoreResult<()> {
    #[cfg(not(feature = "parallel"))]
    {
        println!("This example requires the 'parallel' feature to be enabled.");
        println!("Run with: cargo run --example advanced_parallel_processing --features parallel");
        return Ok(());
    }

    #[cfg(feature = "parallel")]
    {
        println!("=== Advanced Parallel Processing Examples ===\n");

        // Example 1: Custom partitioning for different data distributions
        example_custom_partitioning()?;

        // Example 2: Work-stealing scheduler
        example_work_stealing_scheduler()?;

        // Example 3: Nested parallelism
        example_nested_parallelism()?;

        // Example 4: Load balancing
        example_load_balancing()?;

        // Example 5: Complex nested computation
        example_complex_nested_computation()?;
    }

    Ok(())
}

/// Example 1: Demonstrate custom partitioning strategies
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn example_custom_partitioning() -> CoreResult<()> {
    println!("Example 1: Custom Partitioning Strategies");
    println!("{}", "-".repeat(40));

    // Create different data distributions
    let uniform_data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    let skewed_data: Vec<f64> = (0..10000)
        .map(|i| if i < 1000 { 1.0 } else { (i as f64).powf(2.0) })
        .collect();
    let gaussian_data: Vec<f64> = (0..10000)
        .map(|i| {
            let x = (i as f64 - 5000.0) / 1000.0;
            (-x * x / 2.0).exp()
        })
        .collect();

    // Create partitioner
    let config = PartitionerConfig {
        num_partitions: 4,
        enable_load_balancing: true,
        ..Default::default()
    };
    let partitioner = DataPartitioner::<f64>::new(config);

    // Analyze and partition uniform data
    println!("Uniform distribution:");
    let uniform_dist = partitioner.analyze_distribution(&uniform_data);
    println!("  Detected: {:?}", uniform_dist);
    let uniform_strategy = partitioner.create_strategy(&uniform_dist, uniform_data.len())?;
    let uniform_partitions = partitioner.partition(&uniform_data, &uniform_strategy)?;
    print_partition_info(&uniform_partitions);

    // Analyze and partition skewed data
    println!("\nSkewed distribution:");
    let skewed_dist = partitioner.analyze_distribution(&skewed_data);
    println!("  Detected: {:?}", skewed_dist);
    let skewed_strategy = partitioner.create_strategy(&skewed_dist, skewed_data.len())?;
    let skewed_partitions = partitioner.partition(&skewed_data, &skewed_strategy)?;
    print_partition_info(&skewed_partitions);

    // Analyze and partition Gaussian data
    println!("\nGaussian distribution:");
    let gaussian_dist = partitioner.analyze_distribution(&gaussian_data);
    println!("  Detected: {:?}", gaussian_dist);
    let gaussian_strategy = partitioner.create_strategy(&gaussian_dist, gaussian_data.len())?;
    let gaussian_partitions = partitioner.partition(&gaussian_data, &gaussian_strategy)?;
    print_partition_info(&gaussian_partitions);

    println!();
    Ok(())
}

/// Example 2: Demonstrate work-stealing scheduler
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn example_work_stealing_scheduler() -> CoreResult<()> {
    println!("Example 2: Work-Stealing Scheduler");
    println!("{}", "-".repeat(40));

    // Create scheduler configuration
    let config = SchedulerConfigBuilder::new()
        .workers(4)
        .enable_stealing_heuristics(true)
        .enable_priorities(true)
        .adaptive(true)
        .build();

    let mut scheduler = WorkStealingScheduler::new(config);

    // Submit tasks with different priorities
    let start = Instant::now();

    // For simplicity, use submit_fn which doesn't support priorities directly
    // In a real implementation, you would implement the Task trait
    for i in 0..5 {
        let task = move || -> Result<(), scirs2_core::error::CoreError> {
            println!("Task {} executing", i);
            std::thread::sleep(Duration::from_millis(50));
            Ok(())
        };
        scheduler.submit_fn(task);
    }

    // More tasks
    for i in 5..15 {
        let task = move || -> Result<(), scirs2_core::error::CoreError> {
            println!("Task {} executing", i);
            std::thread::sleep(Duration::from_millis(100));
            Ok(())
        };
        scheduler.submit_fn(task);
    }

    // Wait for completion
    std::thread::sleep(Duration::from_secs(2));

    let stats = scheduler.stats();
    println!("\nScheduler Statistics:");
    println!("  Tasks submitted: {}", stats.tasks_submitted);
    println!("  Tasks completed: {}", stats.tasks_completed);
    println!("  Tasks stolen: {}", stats.successful_steals);
    println!("  Failed steals: {}", stats.failed_steals);
    println!("  Execution time: {:?}", start.elapsed());

    scheduler.shutdown();
    println!();
    Ok(())
}

/// Example 3: Demonstrate nested parallelism
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn example_nested_parallelism() -> CoreResult<()> {
    println!("Example 3: Nested Parallelism");
    println!("{}", "-".repeat(40));

    // Configure resource limits
    let limits = ResourceLimits {
        max_total_threads: 8,
        max_nesting_depth: 3,
        threads_per_level: vec![4, 2, 1],
        ..Default::default()
    };

    // Execute with nested parallelism
    let result = nested_scope_with_limits(limits, |outer_scope| {
        println!("Outer level (depth 0) - starting nested processing");

        // Parallel processing at outer level
        let outer_data: Vec<i32> = (0..100).collect();
        let outer_results = outer_scope.par_iter(outer_data, |x| {
            // Nested parallel operation
            nested_scope(|inner_scope| {
                println!("  Inner level (depth 1) - processing item {}", x);

                let inner_data: Vec<i32> = (0..10).collect();
                let inner_sum: i32 = inner_scope
                    .par_iter(inner_data, |_y| {
                        // Deeply nested operation
                        nested_scope(|deep_scope| {
                            let deep_data: Vec<i32> = (0..5).collect();
                            deep_scope.par_iter(deep_data, |z| z * 2)
                        })
                        .unwrap_or_else(|_| vec![0; 5])
                        .iter()
                        .sum::<i32>()
                    })
                    .unwrap_or_else(|_| vec![0; 10])
                    .iter()
                    .sum();

                Ok(x * inner_sum)
            })
            .unwrap_or(0)
        })?;

        println!("\nOuter results computed: {} items", outer_results.len());
        Ok(outer_results)
    })?;

    println!("Total result items: {}", result.len());
    println!();
    Ok(())
}

/// Example 4: Demonstrate load balancing
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn example_load_balancing() -> CoreResult<()> {
    println!("Example 4: Load Balancing");
    println!("{}", "-".repeat(40));

    // Create load balancer
    let mut balancer = LoadBalancer::new(4, 1.2);

    // Simulate workload execution with varying times
    let partition_times = [
        vec![100, 95, 105, 98],   // Partition 0: consistent ~100ms
        vec![200, 190, 210, 195], // Partition 1: consistent ~200ms
        vec![150, 145, 155, 148], // Partition 2: consistent ~150ms
        vec![50, 55, 45, 52],     // Partition 3: consistent ~50ms
    ];

    println!("Initial execution times (ms):");
    for (i, times) in partition_times.iter().enumerate() {
        println!("  Partition {}: {:?}", i, times);
        for &time in times {
            balancer.recordexecution_time(i, Duration::from_millis(time));
        }
    }

    // Rebalance
    let new_weights = balancer.rebalance();
    println!("\nRebalanced weights:");
    for (i, weight) in new_weights.iter().enumerate() {
        println!("  Partition {}: {:.2}", i, weight);
    }

    let imbalance = balancer.get_imbalance_factor();
    println!("\nLoad imbalance factor: {:.2}", imbalance);

    println!();
    Ok(())
}

/// Example 5: Complex nested computation with different policies
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn example_complex_nested_computation() -> CoreResult<()> {
    println!("Example 5: Complex Nested Computation");
    println!("{}", "-".repeat(40));

    // Test different nested parallelism policies
    let policies = vec![
        (NestedPolicy::Allow, "Allow"),
        (NestedPolicy::Sequential, "Sequential"),
        (NestedPolicy::Deny, "Deny"),
    ];

    for (policy, name) in policies {
        println!("\nTesting with {} policy:", name);

        let config = NestedConfig {
            policy,
            limits: ResourceLimits {
                max_total_threads: 4,
                max_nesting_depth: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let start = Instant::now();
        let result = with_nested_policy(config, || {
            // Simulate matrix multiplication with nested loops
            let size = 50;
            let mut result = 0i64;

            // Outer parallel loop
            let outer_vec: Vec<usize> = (0..size).collect();
            let outer_sum: i64 = outer_vec
                .into_par_iter()
                .map(|i| {
                    // Middle parallel loop (may be sequential based on policy)
                    let middle_vec: Vec<usize> = (0..size).collect();
                    let middle_sum: i64 = middle_vec
                        .into_par_iter()
                        .map(|j| {
                            // Inner loop (always sequential at this depth)
                            let inner_sum: i64 = (0..size).map(|k| (i * j * k) as i64).sum();
                            inner_sum
                        })
                        .sum();
                    middle_sum
                })
                .sum();

            result += outer_sum;
            Ok(result)
        });

        match result {
            Ok(value) => {
                println!("  Result: {}", value);
                println!("  Time: {:?}", start.elapsed());
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    println!();
    Ok(())
}

/// Helper function to print partition information
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn print_partition_info<T>(partitions: &[Vec<T>]) {
    println!("  Partition sizes:");
    for (i, partition) in partitions.iter().enumerate() {
        println!("    Partition {}: {} elements", i, partition.len());
    }

    // Calculate load balance
    if !partitions.is_empty() {
        let sizes: Vec<usize> = partitions.iter().map(|p| p.len()).collect();
        let minsize = *sizes.iter().min().unwrap_or(&0);
        let maxsize = *sizes.iter().max().unwrap_or(&0);
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();

        println!(
            "  Load balance: min={}, max={}, avg={}",
            minsize, maxsize, avg_size
        );
        if minsize > 0 {
            println!("  Imbalance factor: {:.2}", maxsize as f64 / minsize as f64);
        }
    }
}
