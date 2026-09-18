use scirs2_core::profiling::{Profiler, Timer};
use std::thread;
use std::time::Duration;

#[allow(dead_code)]
fn main() {
    println!("Profiling System Example");

    // Only run the example if the profiling feature is enabled
    #[cfg(feature = "profiling")]
    {
        // Start the global profiler
        Profiler::global().lock().expect("Operation failed").start();

        println!("\n--- Basic Timing Example ---");
        basic_timing_example();

        println!("\n--- Function Timing Example ---");
        function_timing_example();

        println!("\n--- Hierarchical Timing Example ---");
        hierarchical_timing_example();

        println!("\n--- Memory Tracking Example ---");
        memory_tracking_example();

        // Print profiling report
        println!("\n--- Profiling Report ---");
        Profiler::global()
            .lock()
            .expect("Operation failed")
            .print_report();

        // Stop the profiler
        Profiler::global().lock().expect("Operation failed").stop();
    }

    #[cfg(not(feature = "profiling"))]
    println!(
        "Profiling feature not enabled. Run with --features=\"profiling\" to see the example."
    );
}

#[cfg(feature = "profiling")]
#[allow(dead_code)]
fn basic_timing_example() {
    // Time a block of code using a timer
    let timer = Timer::start("basic_operation");

    // Simulate some work
    println!("Performing a basic operation...");
    thread::sleep(Duration::from_millis(500));

    // Stop the timer
    timer.stop();
    println!("Basic operation completed");
}

#[cfg(feature = "profiling")]
#[allow(dead_code)]
fn function_timing_example() {
    // Use Timer::time_function to time a function call
    let result = Timer::time_function("calculate_result", || {
        println!("Calculating result...");
        thread::sleep(Duration::from_millis(300));
        42 // Return a result
    });

    println!("Function result: {}", result);
}

#[cfg(feature = "profiling")]
#[allow(dead_code)]
fn hierarchical_timing_example() {
    // Create a parent timer
    let parent_timer = Timer::start("parent_operation");

    println!("Starting parent operation...");
    thread::sleep(Duration::from_millis(200));

    // Create child timers inside the parent
    {
        let child1 = Timer::start_with_parent("child_operation_1", "parent_operation");
        println!("  Performing child operation 1...");
        thread::sleep(Duration::from_millis(300));
        child1.stop();
    }

    {
        let child2 = Timer::start_with_parent("child_operation_2", "parent_operation");
        println!("  Performing child operation 2...");
        thread::sleep(Duration::from_millis(400));
        child2.stop();
    }

    // Finish the parent operation
    thread::sleep(Duration::from_millis(100));
    parent_timer.stop();
    println!("Parent operation completed");
}

#[cfg(feature = "profiling")]
#[allow(dead_code)]
fn memory_tracking_example() {
    // Create some example operations with memory tracking
    {
        let mem_tracker = scirs2_core::profiling::MemoryTracker::start("allocate_vector");
        println!("Allocating a vector...");

        // Allocate a large vector (this is an example of what would be tracked)
        let large_vector = vec![0; 1_000_000];
        println!("Vector size: {} elements", large_vector.len());

        // In a real implementation, the MemoryTracker would detect this allocation automatically
        // In this example, we manually stop the tracker
        mem_tracker.stop();
    }

    {
        let mem_tracker = scirs2_core::profiling::MemoryTracker::start("allocatematrix");
        println!("Allocating a matrix...");

        // Allocate a matrix (this is an example of what would be tracked)
        let matrix_size = 500;
        let matrix = vec![vec![0.0; matrix_size]; matrix_size];
        println!("Matrix size: {}x{} elements", matrix.len(), matrix[0].len());

        // In a real implementation, the MemoryTracker would detect this allocation automatically
        // In this example, we manually stop the tracker
        mem_tracker.stop();
    }

    // Report memory usage - the Profiler will contain the report, which is printed at the end of main
}
