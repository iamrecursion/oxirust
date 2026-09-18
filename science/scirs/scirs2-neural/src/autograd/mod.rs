//! Automatic differentiation module for neural networks.
//!
//! This module provides automatic differentiation capabilities for neural networks,
//! using the autograd crate. This is a simple wrapper for demonstration, showing how
//! we can integrate with autograd.

// We re-export the autograd crate (disabled in minimal version)
// pub extern crate autograd;
/// Example function showing how autograd can be used
#[allow(dead_code)]
pub fn autograd_example() {
    // This demonstrates basic usage of autograd for a simple neural network operation
    println!("Note: Autograd is not included in the minimal version of scirs2-neural.");
    println!("To use autograd, add it as a dependency to your Cargo.toml:");
    println!("autograd = \"1.0\"");
    println!();
    println!("Then import it directly in your code:");
    println!("use autograd::prelude::*;");
    println!();
    println!("For more examples, see:");
    println!("https://docs.rs/autograd/latest/autograd/");
    println!("This integration is provided as a simple demonstration of how to use");
    println!("autograd with SciRS2-neural.");
}
