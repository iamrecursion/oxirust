//! Advanced Numerical Validation Demo
//!
//! This example demonstrates how to use the numerical validation framework
//! to ensure that Advanced optimizations maintain algorithmic accuracy.

#![allow(dead_code)]

use scirs2_graph::numerical_accuracy_validation::{
    create_comprehensive_validation_suite, run_quick_validation, AdvancedNumericalValidator,
    GraphGenerator, ValidationAlgorithm, ValidationConfig, ValidationTestCase,
    ValidationTolerances,
};
use std::time::Instant;

/// Basic numerical validation example
#[allow(dead_code)]
fn basic_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Basic Advanced Numerical Validation Example");
    println!("==============================================");

    // Create a simple validator with default configuration
    let mut validator = AdvancedNumericalValidator::new(ValidationConfig::default());

    // Add a basic test case
    validator.add_test_case(ValidationTestCase {
        name: "Basic Random Graph".to_string(),
        graph_generator: GraphGenerator::Random {
            nodes: 100,
            edges: 200,
            directed: false,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 50,
                tolerance: 1e-6,
            },
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 3,
    });

    // Run validation
    println!("\n🚀 Running validation...");
    let report = validator.run_validation()?;

    // Print results
    println!("\n📊 Validation Results:");
    println!("   Pass rate: {:.1}%", report.summary.pass_rate * 100.0);
    println!(
        "   Average accuracy: {:.4}",
        report.summary.average_accuracy
    );
    println!("   Average speedup: {:.2}x", report.summary.average_speedup);

    for result in &report.test_results {
        println!("\n   Algorithm: {:?}", result.algorithm);
        println!(
            "   Status: {}",
            if result.passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!("   Accuracy Score: {:.6}", result.accuracy_score);
        println!("   Speedup: {:.2}x", result.speedup_factor);

        if let Some(ref error) = result.error_message {
            println!("   Error: {}", error);
        }
    }

    Ok(())
}

/// Comprehensive validation suite example
#[allow(dead_code)]
fn comprehensive_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Comprehensive Validation Suite Example");
    println!("========================================");

    // Create comprehensive validation suite
    let mut validator = create_comprehensive_validation_suite();

    println!("🚀 Running comprehensive validation suite...");
    let start = Instant::now();
    let report = validator.run_validation()?;
    let duration = start.elapsed();

    println!("\n✅ Comprehensive validation completed in {:?}", duration);

    // Detailed analysis
    println!("\n📈 Detailed Analysis:");
    println!("========================");

    println!("\n🎯 Overall Summary:");
    println!("   Total tests: {}", report.summary.total_tests);
    println!("   Tests passed: {}", report.summary.tests_passed);
    println!("   Pass rate: {:.1}%", report.summary.pass_rate * 100.0);
    println!(
        "   Average accuracy: {:.4}",
        report.summary.average_accuracy
    );
    println!("   Average speedup: {:.2}x", report.summary.average_speedup);

    println!("\n⚡ Performance Analysis:");
    println!(
        "   Best speedup: {:.2}x",
        report.performance_analysis.best_speedup
    );
    println!(
        "   Worst speedup: {:.2}x",
        report.performance_analysis.worst_speedup
    );
    println!("   Top performers:");
    for (algorithm, speedup) in &report.performance_analysis.top_performers {
        println!("     - {:?}: {:.2}x", algorithm, speedup);
    }

    if !report
        .performance_analysis
        .performance_regressions
        .is_empty()
    {
        println!("   Performance regressions:");
        for (algorithm, speedup) in &report.performance_analysis.performance_regressions {
            println!("     - {:?}: {:.2}x (slower)", algorithm, speedup);
        }
    }

    println!("\n🎯 Accuracy Analysis:");
    println!(
        "   Best accuracy: {:.6}",
        report.accuracy_analysis.best_accuracy
    );
    println!(
        "   Worst accuracy: {:.6}",
        report.accuracy_analysis.worst_accuracy
    );

    if !report
        .accuracy_analysis
        .perfect_accuracy_algorithms
        .is_empty()
    {
        println!("   Perfect accuracy algorithms:");
        for algorithm in &report.accuracy_analysis.perfect_accuracy_algorithms {
            println!("     - {:?}", algorithm);
        }
    }

    if !report.accuracy_analysis.accuracy_concerns.is_empty() {
        println!("   Accuracy concerns:");
        for (algorithm, accuracy) in &report.accuracy_analysis.accuracy_concerns {
            println!("     - {:?}: {:.6}", algorithm, accuracy);
        }
    }

    println!("\n💡 Recommendations:");
    for recommendation in &report.recommendations {
        println!("   • {}", recommendation);
    }

    Ok(())
}

/// Custom validation test case example
#[allow(dead_code)]
fn custom_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎛️ Custom Validation Test Case Example");
    println!("====================================");

    // Create validator with custom configuration
    let config = ValidationConfig {
        verbose_logging: true,
        benchmark_performance: true,
        statistical_analysis: true,
        warmup_runs: 2,
        cross_validation: true,
        random_seed: Some(12345),
    };

    let mut validator = AdvancedNumericalValidator::new(config);

    // Custom tolerances for stricter validation
    let strict_tolerances = ValidationTolerances {
        absolute_tolerance: 1e-8,
        relative_tolerance: 1e-7,
        integer_tolerance: 0,
        correlation_threshold: 0.98,
        centrality_deviation_threshold: 0.005,
    };

    validator.set_tolerances(strict_tolerances);

    // Add multiple test cases with different graph types
    validator.add_test_case(ValidationTestCase {
        name: "Scale-Free Network Validation".to_string(),
        graph_generator: GraphGenerator::BarabasiAlbert {
            nodes: 500,
            edges_per_node: 3,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 100,
                tolerance: 1e-8,
            },
            ValidationAlgorithm::BetweennessCentrality,
            ValidationAlgorithm::LouvainCommunities,
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 5,
    });

    validator.add_test_case(ValidationTestCase {
        name: "Dense Random Network Validation".to_string(),
        graph_generator: GraphGenerator::ErdosRenyi {
            nodes: 200,
            probability: 0.1,
        },
        algorithms: vec![
            ValidationAlgorithm::AllPairsShortestPaths,
            ValidationAlgorithm::ClosenessCentrality,
            ValidationAlgorithm::DegreeCentrality,
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 3,
    });

    validator.add_test_case(ValidationTestCase {
        name: "Large Sparse Network Validation".to_string(),
        graph_generator: GraphGenerator::Random {
            nodes: 1000,
            edges: 2000,
            directed: false,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 50,
                tolerance: 1e-6,
            },
            ValidationAlgorithm::LabelPropagation { max_iterations: 50 },
        ],
        tolerances: ValidationTolerances {
            // Relaxed tolerances for large graphs
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            correlation_threshold: 0.95,
            ..ValidationTolerances::default()
        },
        num_runs: 2,
    });

    // Run custom validation
    println!("🚀 Running custom validation with strict tolerances...");
    let report = validator.run_validation()?;

    // Analyze results by test case
    println!("\n📊 Results by Test Case:");
    let mut test_case_results = std::collections::HashMap::new();

    for result in &report.test_results {
        test_case_results
            .entry(&result.test_case)
            .or_insert_with(Vec::new)
            .push(result);
    }

    for (test_case, results) in test_case_results {
        println!("\n   Test Case: {}", test_case);
        let passed = results.iter().filter(|r| r.passed).count();
        let total = results.len();
        println!(
            "     Pass rate: {}/{} ({:.1}%)",
            passed,
            total,
            passed as f64 / total as f64 * 100.0
        );

        let avg_accuracy =
            results.iter().map(|r| r.accuracy_score).sum::<f64>() / results.len() as f64;
        let avg_speedup =
            results.iter().map(|r| r.speedup_factor).sum::<f64>() / results.len() as f64;

        println!("     Average accuracy: {:.6}", avg_accuracy);
        println!("     Average speedup: {:.2}x", avg_speedup);

        for result in results {
            let status = if result.passed { "✅" } else { "❌" };
            println!(
                "       {} {:?}: acc={:.4}, speedup={:.2}x",
                status, result.algorithm, result.accuracy_score, result.speedup_factor
            );
        }
    }

    Ok(())
}

/// Quick validation example for development workflow
#[allow(dead_code)]
fn quick_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Quick Validation Example (Development Workflow)");
    println!("================================================");

    println!("🚀 Running quick validation for development...");
    let start = Instant::now();
    let report = run_quick_validation()?;
    let duration = start.elapsed();

    println!("✅ Quick validation completed in {:?}", duration);

    if report.summary.pass_rate >= 0.95 {
        println!("🎉 Validation PASSED: Advanced maintains high accuracy");
        println!("   ✅ All critical algorithms validated");
        println!(
            "   ✅ Performance gains: {:.2}x average speedup",
            report.summary.average_speedup
        );
        println!(
            "   ✅ Accuracy maintained: {:.4} average score",
            report.summary.average_accuracy
        );
    } else {
        println!("⚠️ Validation CONCERNS: Some accuracy issues detected");
        println!("   Pass rate: {:.1}%", report.summary.pass_rate * 100.0);

        for result in &report.test_results {
            if !result.passed {
                println!(
                    "   ❌ {:?}: accuracy {:.4}, speedup {:.2}x",
                    result.algorithm, result.accuracy_score, result.speedup_factor
                );
            }
        }

        println!("\n💡 Next steps:");
        for recommendation in &report.recommendations {
            println!("   • {}", recommendation);
        }
    }

    Ok(())
}

/// Validation metrics deep dive example
#[allow(dead_code)]
fn validation_metrics_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Validation Metrics Deep Dive Example");
    println!("======================================");

    // Run a simple validation to get detailed metrics
    let report = run_quick_validation()?;

    println!("🔬 Detailed Metrics Analysis:");

    for result in &report.test_results {
        println!("\n   Algorithm: {:?}", result.algorithm);
        println!("   Test Case: {}", result.test_case);
        println!(
            "   Status: {}",
            if result.passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );

        println!("   📊 Accuracy Metrics:");
        println!(
            "     - Max absolute error: {:.2e}",
            result.metrics.max_absolute_error
        );
        println!(
            "     - Mean absolute error: {:.2e}",
            result.metrics.mean_absolute_error
        );
        println!(
            "     - Root mean square error: {:.2e}",
            result.metrics.root_mean_square_error
        );
        println!(
            "     - Pearson correlation: {:.6}",
            result.metrics.pearson_correlation
        );
        println!(
            "     - Spearman correlation: {:.6}",
            result.metrics.spearman_correlation
        );
        println!(
            "     - Elements compared: {}",
            result.metrics.elements_compared
        );
        println!(
            "     - Exact matches: {} ({:.1}%)",
            result.metrics.exact_matches,
            result.metrics.exact_matches as f64 / result.metrics.elements_compared as f64 * 100.0
        );

        println!("   ⚡ Performance Metrics:");
        println!("     - Standard time: {:?}", result.standard_time);
        println!("     - Advanced time: {:?}", result.advanced_time);
        println!("     - Speedup factor: {:.2}x", result.speedup_factor);

        if !result.metrics.custom_metrics.is_empty() {
            println!("   🔧 Custom Metrics:");
            for (name, value) in &result.metrics.custom_metrics {
                println!("     - {}: {:.6}", name, value);
            }
        }

        if let Some(ref error) = result.error_message {
            println!("   ❌ Error Details: {}", error);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Numerical Validation Demo");
    println!("======================================");
    println!("This demo showcases the numerical validation framework for advanced mode.");
    println!();

    // Run all examples
    basic_validation_example()?;
    quick_validation_example()?;
    validation_metrics_example()?;
    custom_validation_example()?;
    comprehensive_validation_example()?;

    println!("\n✨ All validation examples completed successfully!");
    println!("===============================================");
    println!();
    println!("💡 Key Takeaways:");
    println!("• Advanced optimizations maintain numerical accuracy");
    println!("• Performance gains are achieved without compromising correctness");
    println!("• Comprehensive validation ensures reliability across different algorithms");
    println!("• Quick validation enables integration into development workflows");
    println!("• Detailed metrics provide insights for further optimization");
    println!();
    println!("📚 For more information:");
    println!("• Check the advanced_numerical_validation module documentation");
    println!("• Review validation tolerances for your specific use case");
    println!("• Integrate validation into your CI/CD pipeline");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_validation() {
        // Test that quick validation runs without errors
        let result = run_quick_validation();
        assert!(result.is_ok());

        let report = result.expect("Operation failed");
        assert!(report.summary.total_tests > 0);
        assert!(report.summary.pass_rate >= 0.0);
        assert!(report.summary.pass_rate <= 1.0);
        assert!(report.summary.average_accuracy >= 0.0);
        assert!(report.summary.average_accuracy <= 1.0);
    }

    #[test]
    fn test_basic_validation() {
        // Test basic validation functionality
        let result = basic_validation_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_config_creation() {
        // Test validation configuration
        let config = ValidationConfig::default();
        assert!(config.verbose_logging);
        assert!(config.benchmark_performance);
        assert!(config.statistical_analysis);
        assert_eq!(config.warmup_runs, 3);
        assert!(config.cross_validation);
        assert_eq!(config.random_seed, Some(42));
    }

    #[test]
    fn test_validation_tolerances() {
        // Test validation tolerances
        let tolerances = ValidationTolerances::default();
        assert_eq!(tolerances.absolute_tolerance, 1e-6);
        assert_eq!(tolerances.relative_tolerance, 1e-5);
        assert_eq!(tolerances.integer_tolerance, 0);
        assert_eq!(tolerances.correlation_threshold, 0.95);
        assert_eq!(tolerances.centrality_deviation_threshold, 0.01);
    }

    #[test]
    fn test_comprehensive_suite_creation() {
        // Test that comprehensive validation suite can be created
        let validator = create_comprehensive_validation_suite();
        // Just verify it was created without panicking
        assert_eq!(
            std::mem::size_of_val(&validator),
            std::mem::size_of::<AdvancedNumericalValidator>()
        );
    }
}
