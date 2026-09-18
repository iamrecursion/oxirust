//! Example demonstrating the pattern recognition system for memory access optimization
//!
//! This example shows how to use the pattern recognition system to detect
//! and optimize memory access patterns in scientific computing workloads.

use scirs2_core::memory_efficient::{
    ComplexPattern, Confidence, PatternRecognitionConfig, PatternRecognizer,
};

#[allow(dead_code)]
fn main() {
    println!("=== Pattern Recognition Example ===\n");

    // Example 1: Row-major pattern detection
    example_row_major_detection();

    // Example 2: Column-major pattern detection
    example_column_major_detection();

    // Example 3: Zigzag pattern detection
    example_zigzag_detection();

    // Example 4: Block pattern detection
    example_block_detection();

    // Example 5: Stencil pattern detection
    example_stencil_detection();

    // Example 6: Real-world matrix multiplication
    examplematrix_multiplication();
}

#[allow(dead_code)]
fn example_row_major_detection() {
    println!("1. Row-Major Pattern Detection");
    println!("------------------------------");

    let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
    recognizer.set_dimensions(vec![8, 8]);

    // Simulate row-major access (sequential)
    println!("Accessing matrix in row-major order...");
    for i in 0..64 {
        recognizer.record_access(i);
    }

    // Check detected patterns
    if let Some(best_pattern) = recognizer.get_best_pattern() {
        println!("Detected pattern: {:?}", best_pattern.pattern_type);
        println!("Confidence: {:?}", best_pattern.confidence);
        println!("Basic pattern: {:?}", recognizer.get_basic_pattern());
    }

    println!();
}

#[allow(dead_code)]
fn example_column_major_detection() {
    println!("2. Column-Major Pattern Detection");
    println!("---------------------------------");

    let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
    recognizer.set_dimensions(vec![8, 8]);

    // Simulate column-major access
    println!("Accessing matrix in column-major order...");
    for j in 0..8 {
        for i in 0..8 {
            recognizer.record_access(i * 8 + j);
        }
    }

    // Check detected patterns
    if let Some(best_pattern) = recognizer.get_best_pattern() {
        println!("Detected pattern: {:?}", best_pattern.pattern_type);
        println!("Confidence: {:?}", best_pattern.confidence);
        println!("Basic pattern: {:?}", recognizer.get_basic_pattern());
    }

    println!();
}

#[allow(dead_code)]
fn example_zigzag_detection() {
    println!("3. Zigzag Pattern Detection");
    println!("---------------------------");

    let config = PatternRecognitionConfig {
        min_history_size: 10, // Lower threshold for demonstration
        ..Default::default()
    };
    let mut recognizer = PatternRecognizer::new(config);
    recognizer.set_dimensions(vec![8, 8]);

    // Simulate zigzag traversal
    println!("Accessing matrix in zigzag pattern...");
    for row in 0..8 {
        if row % 2 == 0 {
            // Even rows: left to right
            for col in 0..8 {
                recognizer.record_access(row * 8 + col);
            }
        } else {
            // Odd rows: right to left
            for col in (0..8).rev() {
                recognizer.record_access(row * 8 + col);
            }
        }
    }

    // Check detected patterns
    let patterns = recognizer.get_patterns();
    for pattern in patterns {
        println!("Detected pattern: {:?}", pattern.pattern_type);
        println!("Confidence: {:?}", pattern.confidence);
        if !pattern.metadata.is_empty() {
            println!("Metadata: {:?}", pattern.metadata);
        }
    }

    println!();
}

#[allow(dead_code)]
fn example_block_detection() {
    println!("4. Block Pattern Detection");
    println!("--------------------------");

    let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
    recognizer.set_dimensions(vec![16, 16]);

    // Simulate 4x4 block access pattern
    println!("Accessing matrix in 4x4 blocks...");

    // Access first block (0,0)
    for i in 0..4 {
        for j in 0..4 {
            recognizer.record_access(i * 16 + j);
        }
    }

    // Access second block (0,1)
    for i in 0..4 {
        for j in 4..8 {
            recognizer.record_access(i * 16 + j);
        }
    }

    // Access third block (1,0)
    for i in 4..8 {
        for j in 0..4 {
            recognizer.record_access(i * 16 + j);
        }
    }

    // Check detected patterns
    let patterns = recognizer.get_patterns();
    for pattern in patterns {
        if let ComplexPattern::Block {
            block_height,
            block_width,
        } = &pattern.pattern_type
        {
            println!("Detected block pattern: {}x{}", block_height, block_width);
            println!("Confidence: {:?}", pattern.confidence);
            println!("Metadata: {:?}", pattern.metadata);
        }
    }

    println!();
}

#[allow(dead_code)]
fn example_stencil_detection() {
    println!("5. Stencil Pattern Detection");
    println!("----------------------------");

    let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
    recognizer.set_dimensions(vec![10, 10]);

    // Simulate 5-point stencil operations
    println!("Accessing matrix with 5-point stencil pattern...");

    // Access interior points with their stencil
    for i in 1..9 {
        for j in 1..9 {
            let center = i * 10 + j;

            // Access center and 4 neighbors
            recognizer.record_access(center); // Center
            recognizer.record_access(center - 10); // North
            recognizer.record_access(center + 1); // East
            recognizer.record_access(center + 10); // South
            recognizer.record_access(center - 1); // West
        }
    }

    // Check detected patterns
    let patterns = recognizer.get_patterns();
    for pattern in patterns {
        if let ComplexPattern::Stencil { dimensions, radius } = &pattern.pattern_type {
            println!(
                "Detected stencil pattern: {}D with radius {}",
                dimensions, radius
            );
            println!("Confidence: {:?}", pattern.confidence);
        }
    }

    println!();
}

#[allow(dead_code)]
fn examplematrix_multiplication() {
    println!("6. Real-World Example: Matrix Multiplication");
    println!("--------------------------------------------");

    let n = 32; // Matrix size
    let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());

    // Set dimensions for the three matrices involved
    recognizer.set_dimensions(vec![n, n]);

    println!("Simulating matrix multiplication access pattern...");
    println!("C = A × B where each matrix is {}×{}", n, n);

    // Simulate the classic matrix multiplication access pattern
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // Access A[i,k]
                recognizer.record_access(i * n + k);
                // Access B[k,j]
                recognizer.record_access(k * n + j);
                // Access C[i,j]
                recognizer.record_access(i * n + j);
            }
        }
    }

    // Analyze detected patterns
    println!("\nAnalysis Results:");
    println!("-----------------");

    let patterns = recognizer.get_patterns();
    println!("Total patterns detected: {}", patterns.len());

    // Group patterns by confidence
    let high_confidence: Vec<_> = patterns
        .iter()
        .filter(|p| p.confidence == Confidence::High)
        .collect();
    let medium_confidence: Vec<_> = patterns
        .iter()
        .filter(|p| p.confidence == Confidence::Medium)
        .collect();

    println!("High confidence patterns: {}", high_confidence.len());
    println!("Medium confidence patterns: {}", medium_confidence.len());

    // Show pattern details
    for pattern in patterns.iter().take(3) {
        println!("\nPattern: {:?}", pattern.pattern_type);
        println!("  Confidence: {:?}", pattern.confidence);
        println!(
            "  First detected: {:?} ago",
            pattern.first_detected.elapsed()
        );
        println!("  Confirmations: {}", pattern.confirmation_count);
    }

    // Optimization suggestions based on patterns
    println!("\nOptimization Suggestions:");
    println!("-------------------------");

    if patterns
        .iter()
        .any(|p| matches!(p.pattern_type, ComplexPattern::RowMajor))
    {
        println!("- Row-major access detected: Consider tiling for better cache usage");
    }

    if patterns
        .iter()
        .any(|p| matches!(p.pattern_type, ComplexPattern::BlockStrided { .. }))
    {
        println!("- Block strided access detected: Consider loop reordering");
    }

    if patterns
        .iter()
        .any(|p| matches!(p.pattern_type, ComplexPattern::Block { .. }))
    {
        println!("- Block access detected: Current tiling strategy seems effective");
    }

    // Show basic pattern for prefetching
    println!(
        "\nRecommended prefetch strategy: {:?}",
        recognizer.get_basic_pattern()
    );
}
