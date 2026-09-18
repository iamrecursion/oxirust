//! Unit tests for the hyperparameter module.

use super::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{SeedableRng, StdRng};
use std::collections::HashMap;

#[test]
fn test_hyperparam_value() {
    let float_val = HyperparamValue::Float(3.5);
    assert_eq!(float_val.as_float(), Some(3.5));
    assert_eq!(float_val.as_int(), Some(3));
    let int_val = HyperparamValue::Int(42);
    assert_eq!(int_val.as_int(), Some(42));
    assert_eq!(int_val.as_float(), Some(42.0));
    let bool_val = HyperparamValue::Bool(true);
    assert_eq!(bool_val.as_bool(), Some(true));
    let string_val = HyperparamValue::String("test".to_string());
    assert_eq!(string_val.as_string(), Some("test"));
}

#[test]
fn test_hyperparam_space_discrete() {
    let space = HyperparamSpace::discrete(vec![
        HyperparamValue::Float(0.1),
        HyperparamValue::Float(0.01),
    ])
    .expect("unwrap");
    let values = space.grid_values(10);
    assert_eq!(values.len(), 2);
    let mut rng = StdRng::seed_from_u64(42);
    let sampled = space.sample(&mut rng);
    assert!(matches!(sampled, HyperparamValue::Float(_)));
}

#[test]
fn test_hyperparam_space_continuous() {
    let space = HyperparamSpace::continuous(0.0, 1.0).expect("unwrap");
    let values = space.grid_values(5);
    assert_eq!(values.len(), 5);
    let mut rng = StdRng::seed_from_u64(42);
    let sampled = space.sample(&mut rng);
    if let HyperparamValue::Float(v) = sampled {
        assert!((0.0..=1.0).contains(&v));
    } else {
        panic!("Expected Float value");
    }
}

#[test]
fn test_hyperparam_space_log_uniform() {
    let space = HyperparamSpace::log_uniform(1e-4, 1e-1).expect("unwrap");
    let values = space.grid_values(3);
    assert_eq!(values.len(), 3);
    let mut rng = StdRng::seed_from_u64(42);
    let sampled = space.sample(&mut rng);
    if let HyperparamValue::Float(v) = sampled {
        assert!((1e-4..=1e-1).contains(&v));
    } else {
        panic!("Expected Float value");
    }
}

#[test]
fn test_hyperparam_space_int_range() {
    let space = HyperparamSpace::int_range(1, 10).expect("unwrap");
    let values = space.grid_values(5);
    assert!(!values.is_empty());
    let mut rng = StdRng::seed_from_u64(42);
    let sampled = space.sample(&mut rng);
    if let HyperparamValue::Int(v) = sampled {
        assert!((1..=10).contains(&v));
    } else {
        panic!("Expected Int value");
    }
}

#[test]
fn test_hyperparam_space_invalid() {
    assert!(HyperparamSpace::discrete(vec![]).is_err());
    assert!(HyperparamSpace::continuous(1.0, 0.0).is_err());
    assert!(HyperparamSpace::log_uniform(0.0, 1.0).is_err());
    assert!(HyperparamSpace::log_uniform(1.0, 0.5).is_err());
    assert!(HyperparamSpace::int_range(10, 5).is_err());
}

#[test]
fn test_grid_search() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::discrete(vec![
            HyperparamValue::Float(0.1),
            HyperparamValue::Float(0.01),
        ])
        .expect("unwrap"),
    );
    param_space.insert(
        "batch_size".to_string(),
        HyperparamSpace::int_range(16, 64).expect("unwrap"),
    );
    let grid_search = GridSearch::new(param_space, 3);
    let configs = grid_search.generate_configs();
    assert!(!configs.is_empty());
    assert!(configs.len() >= 2);
}

#[test]
fn test_grid_search_results() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::discrete(vec![HyperparamValue::Float(0.1)]).expect("unwrap"),
    );
    let mut grid_search = GridSearch::new(param_space, 3);
    let mut config = HashMap::new();
    config.insert("lr".to_string(), HyperparamValue::Float(0.1));
    grid_search.add_result(HyperparamResult::new(config.clone(), 0.9));
    grid_search.add_result(HyperparamResult::new(config.clone(), 0.95));
    grid_search.add_result(HyperparamResult::new(config, 0.85));
    let best = grid_search.best_result().expect("unwrap");
    assert_eq!(best.score, 0.95);
    let sorted = grid_search.sorted_results();
    assert_eq!(sorted[0].score, 0.95);
    assert_eq!(sorted[1].score, 0.9);
    assert_eq!(sorted[2].score, 0.85);
}

#[test]
fn test_random_search() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::continuous(1e-4, 1e-1).expect("unwrap"),
    );
    param_space.insert(
        "dropout".to_string(),
        HyperparamSpace::continuous(0.0, 0.5).expect("unwrap"),
    );
    let mut random_search = RandomSearch::new(param_space, 10, 42);
    let configs = random_search.generate_configs();
    assert_eq!(configs.len(), 10);
    for config in &configs {
        assert!(config.contains_key("lr"));
        assert!(config.contains_key("dropout"));
    }
}

#[test]
fn test_random_search_results() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::discrete(vec![HyperparamValue::Float(0.1)]).expect("unwrap"),
    );
    let mut random_search = RandomSearch::new(param_space, 5, 42);
    let mut config = HashMap::new();
    config.insert("lr".to_string(), HyperparamValue::Float(0.1));
    random_search.add_result(HyperparamResult::new(config.clone(), 0.8));
    random_search.add_result(HyperparamResult::new(config, 0.9));
    let best = random_search.best_result().expect("unwrap");
    assert_eq!(best.score, 0.9);
    assert_eq!(random_search.results().len(), 2);
}

#[test]
fn test_hyperparam_result_with_metrics() {
    let mut config = HashMap::new();
    config.insert("lr".to_string(), HyperparamValue::Float(0.1));
    let result = HyperparamResult::new(config, 0.95)
        .with_metric("accuracy".to_string(), 0.95)
        .with_metric("loss".to_string(), 0.05);
    assert_eq!(result.score, 0.95);
    assert_eq!(result.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(result.metrics.get("loss"), Some(&0.05));
}

#[test]
fn test_grid_search_empty_space() {
    let grid_search = GridSearch::new(HashMap::new(), 3);
    let configs = grid_search.generate_configs();
    assert_eq!(configs.len(), 1);
    assert!(configs[0].is_empty());
}

#[test]
fn test_grid_search_total_configs() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::discrete(vec![
            HyperparamValue::Float(0.1),
            HyperparamValue::Float(0.01),
        ])
        .expect("unwrap"),
    );
    let grid_search = GridSearch::new(param_space, 3);
    assert_eq!(grid_search.total_configs(), 2);
}

#[test]
fn test_gp_kernel_rbf() {
    let kernel = GpKernel::Rbf {
        sigma: 1.0,
        length_scale: 1.0,
    };
    let x1 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("unwrap");
    let x2 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.5, 0.5]).expect("unwrap");
    let k = kernel.compute_kernel(&x1, &x2);
    assert_eq!(k.shape(), &[2, 2]);
    assert!((k[[0, 0]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_gp_kernel_matern() {
    let kernel = GpKernel::Matern32 {
        sigma: 1.0,
        length_scale: 1.0,
    };
    let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("unwrap");
    let k = kernel.compute_kernel(&x, &x);
    assert!((k[[0, 0]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_gp_fit_and_predict() {
    let kernel = GpKernel::Rbf {
        sigma: 1.0,
        length_scale: 0.5,
    };
    let mut gp = GaussianProcess::new(kernel, 1e-6);
    let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0]).expect("unwrap");
    let y_train = Array1::from_vec(vec![0.0, 0.25, 1.0, 2.25, 4.0]);
    gp.fit(x_train, y_train).expect("unwrap");
    let x_test = Array2::from_shape_vec((2, 1), vec![0.75, 1.25]).expect("unwrap");
    let (means, _stds) = gp.predict(&x_test).expect("unwrap");
    assert_eq!(means.len(), 2);
    assert!(means[0] >= 0.0 && means[0] <= 4.0);
    assert!(means[1] >= 0.0 && means[1] <= 4.0);
}

#[test]
fn test_gp_predict_error_not_fitted() {
    let kernel = GpKernel::default();
    let gp = GaussianProcess::new(kernel, 1e-6);
    let x_test = Array2::from_shape_vec((1, 1), vec![0.5]).expect("unwrap");
    let result = gp.predict(&x_test);
    assert!(result.is_err());
}

#[test]
fn test_gp_fit_dimension_mismatch() {
    let kernel = GpKernel::default();
    let mut gp = GaussianProcess::new(kernel, 1e-6);
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).expect("unwrap");
    let y = Array1::from_vec(vec![0.0, 1.0]);
    let result = gp.fit(x, y);
    assert!(result.is_err());
}

#[test]
fn test_acquisition_function_ei() {
    let acq = AcquisitionFunction::ExpectedImprovement { xi: 0.01 };
    assert!(matches!(
        acq,
        AcquisitionFunction::ExpectedImprovement { .. }
    ));
}

#[test]
fn test_acquisition_function_ucb() {
    let acq = AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 };
    assert!(matches!(
        acq,
        AcquisitionFunction::UpperConfidenceBound { .. }
    ));
}

#[test]
fn test_acquisition_function_pi() {
    let acq = AcquisitionFunction::ProbabilityOfImprovement { xi: 0.01 };
    assert!(matches!(
        acq,
        AcquisitionFunction::ProbabilityOfImprovement { .. }
    ));
}

#[test]
fn test_bayesian_optimization_creation() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::log_uniform(1e-4, 1e-1).expect("unwrap"),
    );
    let bayes_opt = BayesianOptimization::new(param_space, 10, 5, 42);
    assert_eq!(bayes_opt.total_budget(), 15);
    assert_eq!(bayes_opt.current_iteration(), 0);
    assert!(!bayes_opt.is_complete());
}

#[test]
fn test_bayesian_optimization_suggest_initial() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::continuous(0.0, 1.0).expect("unwrap"),
    );
    let mut bayes_opt = BayesianOptimization::new(param_space, 5, 3, 42);
    for _ in 0..3 {
        let config = bayes_opt.suggest().expect("unwrap");
        assert!(config.contains_key("lr"));
        bayes_opt.add_result(HyperparamResult::new(config, 0.5));
    }
    assert_eq!(bayes_opt.current_iteration(), 3);
}

#[test]
fn test_bayesian_optimization_suggest_gp_phase() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "x".to_string(),
        HyperparamSpace::continuous(0.0, 1.0).expect("unwrap"),
    );
    let mut bayes_opt = BayesianOptimization::new(param_space, 5, 2, 42);
    let mut config1 = HashMap::new();
    config1.insert("x".to_string(), HyperparamValue::Float(0.25));
    bayes_opt.add_result(HyperparamResult::new(config1, 0.5));
    let mut config2 = HashMap::new();
    config2.insert("x".to_string(), HyperparamValue::Float(0.75));
    bayes_opt.add_result(HyperparamResult::new(config2, 0.8));
    let config = bayes_opt.suggest().expect("unwrap");
    assert!(config.contains_key("x"));
}

#[test]
fn test_bayesian_optimization_with_acquisition() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::log_uniform(1e-4, 1e-1).expect("unwrap"),
    );
    let bayes_opt = BayesianOptimization::new(param_space, 10, 5, 42)
        .with_acquisition(AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 })
        .with_kernel(GpKernel::Matern32 {
            sigma: 1.0,
            length_scale: 0.5,
        })
        .with_noise(1e-5);
    assert!(bayes_opt.total_budget() == 15);
}

#[test]
fn test_bayesian_optimization_best_result() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "x".to_string(),
        HyperparamSpace::continuous(0.0, 1.0).expect("unwrap"),
    );
    let mut bayes_opt = BayesianOptimization::new(param_space, 5, 2, 42);
    let mut config1 = HashMap::new();
    config1.insert("x".to_string(), HyperparamValue::Float(0.3));
    bayes_opt.add_result(HyperparamResult::new(config1, 0.6));
    let mut config2 = HashMap::new();
    config2.insert("x".to_string(), HyperparamValue::Float(0.7));
    bayes_opt.add_result(HyperparamResult::new(config2, 0.9));
    let best = bayes_opt.best_result().expect("unwrap");
    assert_eq!(best.score, 0.9);
}

#[test]
fn test_bayesian_optimization_is_complete() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "x".to_string(),
        HyperparamSpace::continuous(0.0, 1.0).expect("unwrap"),
    );
    let mut bayes_opt = BayesianOptimization::new(param_space, 2, 1, 42);
    assert!(!bayes_opt.is_complete());
    for i in 0..3 {
        let mut config = HashMap::new();
        config.insert("x".to_string(), HyperparamValue::Float(i as f64 * 0.3));
        bayes_opt.add_result(HyperparamResult::new(config, i as f64 * 0.2));
    }
    assert!(bayes_opt.is_complete());
}

#[test]
fn test_bayesian_optimization_multivariate() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::log_uniform(1e-4, 1e-1).expect("unwrap"),
    );
    param_space.insert(
        "batch_size".to_string(),
        HyperparamSpace::int_range(16, 128).expect("unwrap"),
    );
    param_space.insert(
        "dropout".to_string(),
        HyperparamSpace::continuous(0.0, 0.5).expect("unwrap"),
    );
    let mut bayes_opt = BayesianOptimization::new(param_space, 10, 3, 42);
    let config = bayes_opt.suggest().expect("unwrap");
    assert_eq!(config.len(), 3);
    assert!(config.contains_key("lr"));
    assert!(config.contains_key("batch_size"));
    assert!(config.contains_key("dropout"));
}

#[test]
fn test_bayesian_optimization_discrete_space() {
    let mut param_space = HashMap::new();
    param_space.insert(
        "optimizer".to_string(),
        HyperparamSpace::discrete(vec![
            HyperparamValue::String("adam".to_string()),
            HyperparamValue::String("sgd".to_string()),
            HyperparamValue::String("rmsprop".to_string()),
        ])
        .expect("unwrap"),
    );
    let mut bayes_opt = BayesianOptimization::new(param_space, 5, 2, 42);
    let config = bayes_opt.suggest().expect("unwrap");
    assert!(config.contains_key("optimizer"));
    let optimizer = config.get("optimizer").expect("unwrap");
    assert!(matches!(optimizer, HyperparamValue::String(_)));
}

#[test]
fn test_normal_cdf() {
    let cdf_0 = BayesianOptimization::normal_cdf(0.0);
    assert!((cdf_0 - 0.5).abs() < 1e-4);
    let cdf_pos = BayesianOptimization::normal_cdf(1.96);
    assert!((cdf_pos - 0.975).abs() < 1e-2);
    let cdf_neg = BayesianOptimization::normal_cdf(-1.96);
    assert!((cdf_neg - 0.025).abs() < 1e-2);
}

#[test]
fn test_normal_pdf() {
    let pdf_0 = BayesianOptimization::normal_pdf(0.0);
    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    assert!((pdf_0 - expected).abs() < 1e-6);
    let pdf_pos = BayesianOptimization::normal_pdf(1.0);
    let pdf_neg = BayesianOptimization::normal_pdf(-1.0);
    assert!((pdf_pos - pdf_neg).abs() < 1e-10);
}

#[test]
fn test_erf() {
    assert!((BayesianOptimization::erf(0.0) - 0.0).abs() < 1e-6);
    assert!((BayesianOptimization::erf(1.0) - 0.8427).abs() < 1e-3);
    assert!((BayesianOptimization::erf(-1.0) + 0.8427).abs() < 1e-3);
}
