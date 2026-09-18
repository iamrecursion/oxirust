use scirs2_core::logging::{
    configurelogger, ConsoleLogHandler, LogLevel, Logger, LoggerConfig, ProgressTracker,
};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[allow(dead_code)]
fn main() {
    println!("Logging and Diagnostics Example");

    // Only run the example if the logging feature is enabled
    #[cfg(feature = "logging")]
    {
        // Configure the logger
        configurelogger(LoggerConfig {
            min_level: LogLevel::Debug,
            show_timestamps: true,
            show_modules: true,
            module_levels: std::collections::HashMap::new(),
        });

        // Register a custom console handler with a specific format
        let console_handler = Arc::new(ConsoleLogHandler {
            format: "[{timestamp}] {level} [{module}] {message}".to_string(),
        });
        scirs2_core::logging::set_handler(console_handler);

        // Create loggers for different modules
        let mathlogger = Logger::new("math").with_field("precision", "double");

        let iologger = Logger::new("io").with_field("mode", "async");

        // Log some messages at different levels
        mathlogger.info("Starting calculation");
        mathlogger.debug("Using algorithm: Fast Fourier Transform");

        iologger.info("Opening data file");
        iologger.warn("File size is large, this may take some time");

        // Simulate a long-running operation with progress tracking
        simulate_long_operation();
    }

    #[cfg(not(feature = "logging"))]
    println!("Logging feature not enabled. Run with --features=\"logging\" to see the example.");
}

#[cfg(feature = "logging")]
#[allow(dead_code)]
fn simulate_long_operation() {
    println!("\n--- Progress Tracking Example ---");

    let total_steps = 10;
    let mut progress = ProgressTracker::new("Data Processing", total_steps);

    // Process data in steps
    for i in 1..=total_steps {
        // Simulate work
        thread::sleep(Duration::from_millis(300));

        // Update progress
        progress.update(i);

        // Log detailed information occasionally
        if i % 3 == 0 {
            Logger::new("process")
                .with_field("step", i)
                .with_field("memory_usage", format!("{} MB", 100 + i * 5))
                .debug(&format!("Completed processing step {}/{}", i, total_steps));
        }
    }

    // Mark the operation as complete
    progress.complete();

    // Log final result
    Logger::new("process").info("Data processing completed successfully");
}
