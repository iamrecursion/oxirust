use super::*;

use super::*;
use ::ndarray::Array1;

#[test]
fn test_validator_creation() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    // Test basic properties
    assert!(!validator.config.strict_mode);
    assert_eq!(validator.config.max_depth, 100);
}

#[test]
fn test_array_validation() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config.clone()).expect("Test: operation failed");
    let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let constraints = ArrayValidationConstraints::new()
        .withshape(vec![5])
        .with_fieldname("test_array")
        .check_numeric_quality();

    let result = validator
        .validate_ndarray(&array, &constraints, &config)
        .expect("Test: operation failed");
    assert!(result.is_valid());
}

#[test]
fn test_quality_report_generation() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");
    let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let report = validator
        .generate_quality_report(&array, "test_field")
        .expect("Test: operation failed");

    assert!(report.quality_score > 0.9); // Should be high quality
    assert_eq!(report.metrics.completeness, 1.0); // No missing values
}

#[test]
fn test_cache_management() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    // Test cache clearing
    assert!(validator.clear_cache().is_ok());

    // Test cache stats
    let (size, hit_rate) = validator.get_cache_stats().expect("Test: operation failed");
    assert_eq!(size, 0); // Should be empty after clearing
    assert_eq!(hit_rate, 0.0); // No hits yet
}

#[test]
fn test_json_validation() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    let schema = ValidationSchema::new()
        .name("test_schema")
        .require_field("name", DataType::String)
        .require_field("age", DataType::Integer);

    let valid_data = serde_json::json!({
        "name": "John Doe",
        "age": 30
    });

    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "name": "John Doe"
        // Missing required "age" field
    });

    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
    assert_eq!(result.errors().len(), 1);
}

#[test]
fn test_allowed_values_constraint() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    let schema = ValidationSchema::new()
        .name("test_schema")
        .optional_field("status", DataType::String)
        .add_constraint(
            "status",
            Constraint::AllowedValues(vec![
                "active".to_string(),
                "inactive".to_string(),
                "pending".to_string(),
            ]),
        );

    let valid_data = serde_json::json!({
        "status": "active"
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "status": "deleted"
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
}

#[test]
fn test_precision_constraint() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    let schema = ValidationSchema::new()
        .name("test_schema")
        .optional_field("price", DataType::Float64)
        .add_constraint("price", Constraint::Precision { decimal_places: 2 });

    let valid_data = serde_json::json!({
        "price": 19.99
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "price": 19.999
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
}

#[test]
fn test_array_size_constraint() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    let schema = ValidationSchema::new()
        .name("test_schema")
        .optional_field("tags", DataType::Array(Box::new(DataType::String)))
        .add_constraint("tags", Constraint::ArraySize { min: 1, max: 5 });

    let valid_data = serde_json::json!({
        "tags": ["rust", "programming", "science"]
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "tags": ["too", "many", "tags", "here", "six", "seven"]
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
}

#[test]
fn test_array_elements_constraint() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).expect("Test: operation failed");

    let schema = ValidationSchema::new()
        .name("test_schema")
        .optional_field("scores", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint(
            "scores",
            Constraint::ArrayElements(Box::new(Constraint::Range {
                min: 0.0,
                max: 100.0,
            })),
        );

    let valid_data = serde_json::json!({
        "scores": [85.5, 92.0, 78.3, 95.0]
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "scores": [85.5, 92.0, 105.0, 95.0]
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
}

#[test]

fn test_composite_constraint_validation() {
    let validator = Validator::new(ValidationConfig::default()).expect("Test: operation failed");

    // Test AND constraint
    let schema = ValidationSchema::new()
        .require_field("age", DataType::Float64)
        .add_constraint(
            "age",
            Constraint::And(vec![
                Constraint::Range {
                    min: 0.0,
                    max: 150.0,
                },
                Constraint::NotNull,
            ]),
        );

    let valid_data = serde_json::json!({
        "age": 25.0
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "age": -5.0
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    // Test OR constraint
    let schema = ValidationSchema::new()
        .require_field("status", DataType::String)
        .add_constraint(
            "status",
            Constraint::Or(vec![
                Constraint::Pattern("^active$".to_string()),
                Constraint::Pattern("^inactive$".to_string()),
            ]),
        );

    let valid_data = serde_json::json!({
        "status": "active"
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "status": "pending"
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    // Test NOT constraint
    let schema = ValidationSchema::new()
        .require_field("password", DataType::String)
        .add_constraint(
            "password",
            Constraint::Not(Box::new(Constraint::Pattern("password".to_string()))),
        );

    let valid_data = serde_json::json!({
        "password": "s3cr3t"
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({
        "password": "password123"
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    // Test IF-THEN constraint
    let schema = ValidationSchema::new()
        .optional_field("premium", DataType::Boolean)
        .optional_field("limit", DataType::Float64)
        .add_constraint(
            "limit",
            Constraint::If {
                condition: Box::new(Constraint::NotNull),
                then_constraint: Box::new(Constraint::Range {
                    min: 0.0,
                    max: 1000000.0,
                }),
                else_constraint: None,
            },
        );

    let valid_data = serde_json::json!({
        "premium": true,
        "limit": 50000.0
    });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());
}

#[test]

fn test_edge_case_validations() {
    let validator = Validator::new(ValidationConfig::default()).expect("Test: operation failed");

    // Test empty AND constraint
    let schema = ValidationSchema::new()
        .require_field("value", DataType::Float64)
        .add_constraint("value", Constraint::And(vec![]));

    let data = serde_json::json!({ "value": 42.0 });
    let result = validator
        .validate(&data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid()); // Empty AND should pass

    // Test empty OR constraint
    let schema = ValidationSchema::new()
        .require_field("value", DataType::Float64)
        .add_constraint("value", Constraint::Or(vec![]));

    let result = validator
        .validate(&data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid()); // Empty OR currently passes, but could be considered invalid

    // Test nested AND/OR combinations
    let complex_constraint = Constraint::And(vec![
        Constraint::Or(vec![
            Constraint::Range {
                min: 0.0,
                max: 50.0,
            },
            Constraint::Range {
                min: 100.0,
                max: 150.0,
            },
        ]),
        Constraint::Not(Box::new(Constraint::Range {
            min: 25.0,
            max: 30.0,
        })),
    ]);

    let schema = ValidationSchema::new()
        .require_field("score", DataType::Float64)
        .add_constraint("score", complex_constraint);

    // Value 20 should pass: in range 0-50 AND not in range 25-30
    let valid_data = serde_json::json!({ "score": 20.0 });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    // Value 27 should fail: in range 0-50 BUT also in range 25-30
    let invalid_data = serde_json::json!({ "score": 27.0 });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    // Test IF-THEN-ELSE with field dependencies
    // Note: This test shows a limitation - we need a way to reference other fields
    // For now, we'll test a simpler case where the condition is on the same field
    let schema = ValidationSchema::new()
        .optional_field("value", DataType::Float64)
        .add_constraint(
            "value",
            Constraint::If {
                condition: Box::new(Constraint::Range {
                    min: 1000.0,
                    max: f64::INFINITY,
                }),
                then_constraint: Box::new(Constraint::Range {
                    min: 1000.0,
                    max: 10000.0,
                }),
                else_constraint: Some(Box::new(Constraint::Range {
                    min: 0.0,
                    max: 100.0,
                })),
            },
        );

    // High value (>= 1000) must be in range 1000-10000
    let valid_high = serde_json::json!({
        "value": 5000.0
    });
    let result = validator
        .validate(&valid_high, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    // Low value (< 1000) must be in range 0-100
    let valid_low = serde_json::json!({
        "value": 50.0
    });
    let result = validator
        .validate(&valid_low, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    // High value out of allowed range (should fail)
    let invalid_high = serde_json::json!({
        "value": 15000.0
    });
    let result = validator
        .validate(&invalid_high, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    // Low value out of allowed range (should fail)
    let invalid_low = serde_json::json!({
        "value": 150.0
    });
    let result = validator
        .validate(&invalid_low, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    // Test multiple NOT constraints
    let schema = ValidationSchema::new()
        .require_field("code", DataType::String)
        .add_constraint(
            "code",
            Constraint::And(vec![
                Constraint::Not(Box::new(Constraint::Pattern("test".to_string()))),
                Constraint::Not(Box::new(Constraint::Pattern("debug".to_string()))),
                Constraint::Length { min: 3, max: 10 },
            ]),
        );

    let valid_data = serde_json::json!({ "code": "prod123" });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({ "code": "test123" });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
}

#[test]

fn test_constrainterror_messages() {
    let validator = Validator::new(ValidationConfig::default()).expect("Test: operation failed");

    // Test that OR constraint provides meaningful error
    let schema = ValidationSchema::new()
        .require_field("format", DataType::String)
        .add_constraint(
            "format",
            Constraint::Or(vec![
                Constraint::Pattern("^[A-Z]{3}$".to_string()),
                Constraint::Pattern("^[0-9]{6}$".to_string()),
            ]),
        );

    let invalid_data = serde_json::json!({ "format": "abc123" });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    let errors = result.errors();
    assert!(!errors.is_empty());
    assert!(errors[0]
        .message
        .contains("None of the OR constraints passed"));

    // Test nested constraint error propagation
    let schema = ValidationSchema::new()
        .require_field("data", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint(
            "data",
            Constraint::And(vec![
                Constraint::ArraySize { min: 2, max: 10 },
                Constraint::ArrayElements(Box::new(Constraint::Range {
                    min: 0.0,
                    max: 100.0,
                })),
            ]),
        );

    let invalid_data = serde_json::json!({
        "data": [10.0, 20.0, 150.0] // 150 is out of range
    });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());

    let errors = result.errors();
    assert!(errors.iter().any(|e| e.fieldpath.contains("[2]")));
}

#[test]

fn test_performance_edge_cases() {
    let validator = Validator::new(ValidationConfig::default()).expect("Test: operation failed");

    // Test deeply nested constraints
    let mut constraint = Constraint::Range {
        min: 0.0,
        max: 100.0,
    };
    for _ in 0..10 {
        constraint = Constraint::And(vec![constraint.clone(), Constraint::NotNull]);
    }

    let schema = ValidationSchema::new()
        .require_field("value", DataType::Float64)
        .add_constraint("value", constraint);

    let data = serde_json::json!({ "value": 50.0 });
    let result = validator
        .validate(&data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    // Test large OR constraint
    let many_patterns: Vec<Constraint> = (0..100)
        .map(|i| Constraint::Pattern(format!("pattern{}", i)))
        .collect();

    let schema = ValidationSchema::new()
        .require_field("text", DataType::String)
        .add_constraint("text", Constraint::Or(many_patterns));

    let valid_data = serde_json::json!({ "text": "pattern42" });
    let result = validator
        .validate(&valid_data, &schema)
        .expect("Test: operation failed");
    assert!(result.is_valid());

    let invalid_data = serde_json::json!({ "text": "no-match" });
    let result = validator
        .validate(&invalid_data, &schema)
        .expect("Test: operation failed");
    assert!(!result.is_valid());
}
