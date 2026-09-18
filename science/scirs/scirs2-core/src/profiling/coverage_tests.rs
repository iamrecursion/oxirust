use super::*;

use super::*;

#[test]
fn test_coverage_config_builder() {
    let config = CoverageConfig::development()
        .with_threshold(90.0)
        .with_coverage_types(vec![CoverageType::Line, CoverageType::Branch])
        .with_report_format(ReportFormat::Html);

    assert_eq!(config.coverage_threshold, 90.0);
    assert_eq!(config.coverage_types.len(), 2);
    assert_eq!(config.report_formats, vec![ReportFormat::Html]);
}

#[test]
fn test_coverage_analyzer_creation() {
    let config = CoverageConfig::default();
    let analyzer = CoverageAnalyzer::new(config);
    assert!(analyzer.is_ok());
}

#[test]
fn test_file_coverage_calculations() {
    let mut file_cov = FileCoverage {
        file_path: PathBuf::from("test.rs"),
        total_lines: 100,
        covered_lines: 80,
        line_hits: BTreeMap::new(),
        branches: vec![
            BranchCoverage {
                line_number: 10,
                branch_id: "b1".to_string(),
                true_count: 5,
                false_count: 3,
                branch_type: BranchType::IfElse,
                source_snippet: "if condition".to_string(),
            },
            BranchCoverage {
                line_number: 20,
                branch_id: "b2".to_string(),
                true_count: 0,
                false_count: 0,
                branch_type: BranchType::IfElse,
                source_snippet: "if other".to_string(),
            },
        ],
        functions: vec![FunctionCoverage {
            function_name: "test_fn".to_string(),
            start_line: 5,
            end_line: 15,
            execution_count: 10,
            complexity: 3,
            parameter_count: 2,
            return_complexity: 1,
        }],
        integrations: vec![],
        modified_time: SystemTime::now(),
        collected_at: SystemTime::now(),
    };

    // Add some line hits
    file_cov.line_hits.insert(1, 5);
    file_cov.line_hits.insert(2, 3);
    file_cov.line_hits.insert(10, 8);

    assert_eq!(file_cov.line_coverage_percentage(), 80.0);
    assert_eq!(file_cov.branch_coverage_percentage(), 50.0); // 1 out of 2 branches covered
    assert_eq!(file_cov.function_coverage_percentage(), 100.0); // 1 out of 1 function covered

    let uncovered = file_cov.uncovered_lines();
    assert_eq!(uncovered.len(), 97); // 100 - 3 covered lines

    let hot_spots = file_cov.hot_spots(5);
    assert_eq!(hot_spots.len(), 2); // Lines with >= 5 hits
}

#[test]
fn test_branch_coverage_analysis() {
    let branch = BranchCoverage {
        line_number: 10,
        branch_id: "test_branch".to_string(),
        true_count: 8,
        false_count: 2,
        branch_type: BranchType::IfElse,
        source_snippet: "if x > 0".to_string(),
    };

    assert!(branch.is_covered());
    assert_eq!(branch.total_executions(), 10);
    assert!((branch.balance_score() - 0.4).abs() < f64::EPSILON); // min(8,2) / 10 * 2
}

#[test]
fn test_function_coverage_score() {
    let function = FunctionCoverage {
        function_name: "complex_function".to_string(),
        start_line: 1,
        end_line: 50,
        execution_count: 5,
        complexity: 8,
        parameter_count: 4,
        return_complexity: 2,
    };

    let score = function.coverage_score();
    assert!(score > 0.0 && score <= 1.0);

    // Test uncovered function
    let uncovered_function = FunctionCoverage {
        function_name: "unused_function".to_string(),
        start_line: 60,
        end_line: 70,
        execution_count: 0,
        complexity: 5,
        parameter_count: 2,
        return_complexity: 1,
    };

    assert_eq!(uncovered_function.coverage_score(), 0.0);
}

#[test]
fn test_quality_gate_evaluation() {
    let config = CoverageConfig {
        coverage_threshold: 80.0,
        branch_threshold: 70.0,
        integration_threshold: 60.0,
        ..Default::default()
    };

    let analyzer = CoverageAnalyzer::new(config).expect("Operation failed");

    let stats = CoverageStatistics {
        total_lines: 1000,
        covered_lines: 750,
        line_coverage_percentage: 75.0,
        total_branches: 200,
        covered_branches: 130,
        branch_coverage_percentage: 65.0,
        total_functions: 50,
        covered_functions: 45,
        function_coverage_percentage: 90.0,
        total_integrations: 30,
        covered_integrations: 20,
        integration_coverage_percentage: 66.7,
        files_analyzed: 25,
    };

    let quality_gates = analyzer.evaluate_quality_gates(&stats);

    assert!(!quality_gates.overall_passed);
    assert!(!quality_gates.line_coverage_passed); // 75% < 80%
    assert!(!quality_gates.branch_coverage_passed); // 65% < 70%
    assert!(quality_gates.integration_coverage_passed); // 66.7% > 60%
    assert_eq!(quality_gates.failures.len(), 2);
}

#[test]
fn test_coverage_recommendation_generation() {
    let config = CoverageConfig::default();
    let analyzer = CoverageAnalyzer::new(config).expect("Operation failed");

    let stats = CoverageStatistics {
        total_lines: 1000,
        covered_lines: 600,
        line_coverage_percentage: 60.0,
        total_branches: 200,
        covered_branches: 100,
        branch_coverage_percentage: 50.0,
        total_functions: 50,
        covered_functions: 30,
        function_coverage_percentage: 60.0,
        total_integrations: 30,
        covered_integrations: 15,
        integration_coverage_percentage: 50.0,
        files_analyzed: 25,
    };

    let recommendations = analyzer
        .generate_recommendations(&stats)
        .expect("Operation failed");

    assert!(!recommendations.is_empty());

    // Should have branch coverage recommendation since 50% < 70%
    let has_branch_rec = recommendations
        .iter()
        .any(|r| r.recommendation_type == RecommendationType::ImproveBranchCoverage);
    assert!(has_branch_rec);

    // Check priority ordering
    let priorities: Vec<_> = recommendations.iter().map(|r| r.priority).collect();
    assert!(priorities.windows(2).all(|w| w[0] >= w[1])); // Should be sorted by priority
}

#[test]
fn test_coverage_trends() {
    let config = CoverageConfig::default();
    let analyzer = CoverageAnalyzer::new(config).expect("Operation failed");

    // Add some historical data
    let mut history = analyzer.history.lock().expect("Operation failed");
    let now = SystemTime::now();

    history.push(CoverageDataPoint {
        timestamp: now - Duration::from_secs(7 * 24 * 60 * 60), // 7 days ago
        coverage_percentage: 70.0,
        branch_coverage_percentage: 60.0,
        version: Some("v1.0.0".to_string()),
        test_count: 100,
    });

    history.push(CoverageDataPoint {
        timestamp: now - Duration::from_secs(3 * 24 * 60 * 60), // 3 days ago
        coverage_percentage: 75.0,
        branch_coverage_percentage: 65.0,
        version: Some("v1.1.0".to_string()),
        test_count: 120,
    });

    history.push(CoverageDataPoint {
        timestamp: now,
        coverage_percentage: 80.0,
        branch_coverage_percentage: 70.0,
        version: Some("v1.2.0".to_string()),
        test_count: 150,
    });

    drop(history);

    let trends = analyzer.calculate_trends().expect("Operation failed");
    assert!(trends.is_some());

    let trends = trends.expect("Operation failed");
    assert_eq!(trends.trend_direction, TrendDirection::Improving);
    assert!(trends.change_rate > 0.0); // Positive change rate
    assert!(trends.predicted_coverage.is_some());
}
