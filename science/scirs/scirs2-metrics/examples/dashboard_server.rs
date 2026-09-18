//! Example of using the interactive dashboard with HTTP server
//!
//! This example demonstrates how to create and use the interactive dashboard
//! with a real HTTP server for visualizing machine learning metrics.
//!
//! Run with: cargo run --example dashboard_server --features dashboard_server

#[cfg(feature = "dashboard_server")]
use scirs2_metrics::dashboard::{server::start_http_server, DashboardConfig, InteractiveDashboard};
#[cfg(feature = "dashboard_server")]
use std::{thread, time::Duration};

#[cfg(feature = "dashboard_server")]
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create dashboard configuration
    let config = DashboardConfig {
        title: "ML Training Metrics Dashboard".to_string(),
        refresh_interval: 2, // Refresh every 2 seconds
        ..Default::default()
    };

    // Create interactive dashboard
    let dashboard = InteractiveDashboard::new(config);

    // Add some initial metrics
    dashboard.add_metric("accuracy", 0.75)?;
    dashboard.add_metric("loss", 0.45)?;
    dashboard.add_metric("learning_rate", 0.001)?;

    println!("Starting dashboard server...");

    // Start the HTTP server
    let server = start_http_server(dashboard.clone())?;

    println!("Dashboard is available at http://127.0.0.1:8080");
    println!("Press Ctrl+C to stop the server");

    // Simulate training loop that updates metrics
    let mut epoch = 0;
    loop {
        epoch += 1;

        // Simulate improving accuracy
        let accuracy = 0.75 + (epoch as f64 * 0.01).min(0.24);
        dashboard.add_metric("accuracy", accuracy)?;

        // Simulate decreasing loss
        let loss = 0.45 * (0.95_f64).powi(epoch);
        dashboard.add_metric("loss", loss)?;

        // Simulate learning rate decay
        let lr = 0.001 * (0.99_f64).powi(epoch);
        dashboard.add_metric("learning_rate", lr)?;

        // Add epoch-specific metrics
        dashboard.add_metric("epoch", epoch as f64)?;

        println!("Epoch {epoch}: accuracy={accuracy:.4}, loss={loss:.4}, lr={lr:.6}");

        // Wait before next update
        thread::sleep(Duration::from_secs(3));

        if epoch >= 50 {
            println!("Training completed!");
            break;
        }
    }

    // Keep server running for a while after training
    println!("Keeping dashboard running for viewing results...");
    thread::sleep(Duration::from_secs(30));

    drop(server);
    println!("Dashboard server stopped");

    Ok(())
}

#[cfg(not(feature = "dashboard_server"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'dashboard_server' feature to be enabled.");
    println!("Run with: cargo run --example dashboard_server --features dashboard_server");
}
