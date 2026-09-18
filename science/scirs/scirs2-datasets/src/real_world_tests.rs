use super::*;

use super::*;
use scirs2_core::random::Uniform;

#[test]
fn test_load_titanic() {
    let dataset = load_titanic().expect("Operation failed");
    assert_eq!(dataset.n_samples(), 891);
    assert_eq!(dataset.n_features(), 7);
    assert!(dataset.target.is_some());
}

#[test]
fn test_load_california_housing() {
    let dataset = load_california_housing().expect("Operation failed");
    assert_eq!(dataset.n_samples(), 20640);
    assert_eq!(dataset.n_features(), 8);
    assert!(dataset.target.is_some());
}

#[test]
fn test_load_heart_disease() {
    let dataset = load_heart_disease().expect("Operation failed");
    assert_eq!(dataset.n_samples(), 303);
    assert_eq!(dataset.n_features(), 13);
    assert!(dataset.target.is_some());
}

#[test]
fn test_list_datasets() {
    let datasets = list_real_world_datasets();
    assert!(!datasets.is_empty());
    assert!(datasets.contains(&"titanic".to_string()));
    assert!(datasets.contains(&"california_housing".to_string()));
}

#[test]
fn test_real_world_config() {
    let config = RealWorldConfig {
        use_cache: false,
        download_if_missing: false,
        ..Default::default()
    };

    assert!(!config.use_cache);
    assert!(!config.download_if_missing);
}
