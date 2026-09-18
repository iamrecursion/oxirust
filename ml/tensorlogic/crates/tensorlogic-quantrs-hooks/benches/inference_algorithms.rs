//! Benchmarks for inference algorithms comparison
//!
//! This benchmark suite compares the performance of different inference methods:
//! - Variable Elimination (exact)
//! - Junction Tree (exact)
//! - Sum-Product BP (exact for trees, approximate for loopy graphs)
//! - Mean-Field VI (approximate)
//! - Bethe Approximation (approximate)
//! - Tree-Reweighted BP (approximate with bounds)
//! - Expectation Propagation (approximate)
//! - Gibbs Sampling (MCMC)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array;
use std::hint::black_box;
use tensorlogic_quantrs_hooks::{
    BetheApproximation, ExpectationPropagation, Factor, FactorGraph, GibbsSampler, JunctionTree,
    MeanFieldInference, MessagePassingAlgorithm, SumProductAlgorithm, TreeReweightedBP,
    VariableElimination,
};

/// Create a Bayesian Network: Sprinkler example
/// Rain -> Sprinkler -> Wet Grass <- Rain
fn create_sprinkler_network() -> FactorGraph {
    let mut graph = FactorGraph::new();

    // Variables
    graph.add_variable_with_card("Rain".to_string(), "Weather".to_string(), 2);
    graph.add_variable_with_card("Sprinkler".to_string(), "Device".to_string(), 2);
    graph.add_variable_with_card("WetGrass".to_string(), "State".to_string(), 2);

    // P(Rain)
    let p_rain = Factor::new(
        "P(Rain)".to_string(),
        vec!["Rain".to_string()],
        Array::from_shape_vec(vec![2], vec![0.8, 0.2])
            .expect("create_sprinkler_network: Failed to create P(Rain) array")
            .into_dyn(),
    )
    .expect("create_sprinkler_network: Failed to create P(Rain) factor");
    graph
        .add_factor(p_rain)
        .expect("create_sprinkler_network: Failed to add P(Rain)");

    // P(Sprinkler | Rain)
    let p_sprinkler = Factor::new(
        "P(Sprinkler|Rain)".to_string(),
        vec!["Rain".to_string(), "Sprinkler".to_string()],
        Array::from_shape_vec(vec![2, 2], vec![0.6, 0.4, 0.99, 0.01])
            .expect("create_sprinkler_network: Failed to create P(Sprinkler|Rain) array")
            .into_dyn(),
    )
    .expect("create_sprinkler_network: Failed to create P(Sprinkler|Rain) factor");
    graph
        .add_factor(p_sprinkler)
        .expect("create_sprinkler_network: Failed to add P(Sprinkler|Rain)");

    // P(WetGrass | Sprinkler, Rain)
    let p_wet = Factor::new(
        "P(WetGrass|Sprinkler,Rain)".to_string(),
        vec![
            "Sprinkler".to_string(),
            "Rain".to_string(),
            "WetGrass".to_string(),
        ],
        Array::from_shape_vec(
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.2, 0.8, 0.1, 0.9, 0.01, 0.99],
        )
        .expect("create_sprinkler_network: Failed to create P(WetGrass|...) array")
        .into_dyn(),
    )
    .expect("create_sprinkler_network: Failed to create P(WetGrass|...) factor");
    graph
        .add_factor(p_wet)
        .expect("create_sprinkler_network: Failed to add P(WetGrass|...)");

    graph
}

/// Create a chain MRF for benchmarking
fn create_chain_mrf(length: usize, card: usize) -> FactorGraph {
    let mut graph = FactorGraph::new();

    // Add variables
    for i in 0..length {
        graph.add_variable_with_card(format!("X_{}", i), "Domain".to_string(), card);
    }

    // Add pairwise potentials
    let size = card * card;
    let values: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
    let shape = vec![card, card];

    for i in 0..(length - 1) {
        let array = Array::from_shape_vec(shape.clone(), values.clone())
            .expect("create_chain_mrf: Failed to create factor array")
            .into_dyn();
        let factor = Factor::new(
            format!("psi_{}_{}", i, i + 1),
            vec![format!("X_{}", i), format!("X_{}", i + 1)],
            array,
        )
        .expect("create_chain_mrf: Failed to create factor");
        graph
            .add_factor(factor)
            .expect("create_chain_mrf: Failed to add factor");
    }

    graph
}

/// Create a grid MRF
fn create_grid_mrf(rows: usize, cols: usize, card: usize) -> FactorGraph {
    let mut graph = FactorGraph::new();

    // Add variables
    for i in 0..rows {
        for j in 0..cols {
            graph.add_variable_with_card(format!("X_{}_{}", i, j), "Pixel".to_string(), card);
        }
    }

    // Add pairwise factors
    let size = card * card;
    let values: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
    let shape = vec![card, card];

    // Horizontal edges
    for i in 0..rows {
        for j in 0..(cols - 1) {
            let array = Array::from_shape_vec(shape.clone(), values.clone())
                .expect("create_grid_mrf: Failed to create horizontal edge array")
                .into_dyn();
            let factor = Factor::new(
                format!("h_{}_{}", i, j),
                vec![format!("X_{}_{}", i, j), format!("X_{}_{}", i, j + 1)],
                array,
            )
            .expect("create_grid_mrf: Failed to create horizontal edge factor");
            graph
                .add_factor(factor)
                .expect("create_grid_mrf: Failed to add horizontal edge factor");
        }
    }

    // Vertical edges
    for i in 0..(rows - 1) {
        for j in 0..cols {
            let array = Array::from_shape_vec(shape.clone(), values.clone())
                .expect("create_grid_mrf: Failed to create vertical edge array")
                .into_dyn();
            let factor = Factor::new(
                format!("v_{}_{}", i, j),
                vec![format!("X_{}_{}", i, j), format!("X_{}_{}", i + 1, j)],
                array,
            )
            .expect("create_grid_mrf: Failed to create vertical edge factor");
            graph
                .add_factor(factor)
                .expect("create_grid_mrf: Failed to add vertical edge factor");
        }
    }

    graph
}

/// Benchmark variable elimination
fn bench_variable_elimination(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_elimination");

    let graph = create_sprinkler_network();

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network", |b| {
        b.iter(|| {
            let ve = VariableElimination::new();
            black_box(
                ve.marginalize(&graph, "WetGrass")
                    .expect("VE sprinkler_network marginalize failed"),
            );
        });
    });

    // Chain MRF
    for length in [5, 10, 15] {
        let chain = create_chain_mrf(length, 2);

        group.throughput(Throughput::Elements(length as u64));
        group.bench_with_input(BenchmarkId::new("chain_length", length), &chain, |b, g| {
            b.iter(|| {
                let ve = VariableElimination::new();
                black_box(
                    ve.marginalize(g, "X_0")
                        .expect("VE chain_length marginalize failed"),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark junction tree
fn bench_junction_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("junction_tree");

    let graph = create_sprinkler_network();

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network_build", |b| {
        b.iter(|| {
            black_box(
                JunctionTree::from_factor_graph(&graph).expect("JT sprinkler_network_build failed"),
            );
        });
    });

    group.bench_function("sprinkler_network_calibrate", |b| {
        let mut tree = JunctionTree::from_factor_graph(&graph)
            .expect("JT sprinkler_network_calibrate: from_factor_graph failed");
        b.iter(|| {
            tree.calibrate()
                .expect("JT sprinkler_network_calibrate: calibrate failed");
            black_box(());
        });
    });

    group.bench_function("sprinkler_network_query", |b| {
        let mut tree = JunctionTree::from_factor_graph(&graph)
            .expect("JT sprinkler_network_query: from_factor_graph failed");
        tree.calibrate()
            .expect("JT sprinkler_network_query: calibrate failed");
        b.iter(|| {
            black_box(
                tree.query_marginal("WetGrass")
                    .expect("JT sprinkler_network_query: query_marginal failed"),
            );
        });
    });

    // Chain MRF
    for length in [5, 10, 15] {
        let chain = create_chain_mrf(length, 2);

        group.throughput(Throughput::Elements(length as u64));
        group.bench_with_input(BenchmarkId::new("chain_build", length), &chain, |b, g| {
            b.iter(|| {
                black_box(
                    JunctionTree::from_factor_graph(g)
                        .expect("JT chain_build: from_factor_graph failed"),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark belief propagation
fn bench_belief_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("belief_propagation");

    let graph = create_sprinkler_network();
    let algorithm = SumProductAlgorithm::default();

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network", |b| {
        b.iter(|| {
            black_box(
                algorithm
                    .run(&graph)
                    .expect("BP sprinkler_network run failed"),
            );
        });
    });

    // Chain (tree structure - exact)
    for length in [5, 10, 20] {
        let chain = create_chain_mrf(length, 2);

        group.throughput(Throughput::Elements(length as u64));
        group.bench_with_input(BenchmarkId::new("chain_length", length), &chain, |b, g| {
            b.iter(|| {
                black_box(algorithm.run(g).expect("BP chain_length run failed"));
            });
        });
    }

    // Grid (loopy - approximate)
    for (rows, cols) in [(2, 2), (3, 3), (4, 4)] {
        let grid = create_grid_mrf(rows, cols, 2);
        let loopy_bp = SumProductAlgorithm::new(20, 1e-4, 0.0);

        let num_vars = rows * cols;
        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("grid_loopy", format!("{}x{}", rows, cols)),
            &grid,
            |b, g| {
                b.iter(|| {
                    black_box(loopy_bp.run(g).expect("BP grid_loopy run failed"));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mean-field variational inference
fn bench_mean_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_field");

    let graph = create_sprinkler_network();
    let mf = MeanFieldInference::new(100, 1e-4);

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network", |b| {
        b.iter(|| {
            black_box(mf.run(&graph).expect("MF sprinkler_network run failed"));
        });
    });

    // Grid MRF
    for (rows, cols) in [(2, 2), (3, 3), (4, 4)] {
        let grid = create_grid_mrf(rows, cols, 2);

        let num_vars = rows * cols;
        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", rows, cols)),
            &grid,
            |b, g| {
                b.iter(|| {
                    black_box(mf.run(g).expect("MF grid_size run failed"));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Bethe approximation
fn bench_bethe(c: &mut Criterion) {
    let mut group = c.benchmark_group("bethe_approximation");

    let graph = create_sprinkler_network();
    let bethe = BetheApproximation::new(50, 1e-4, 0.0);

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network", |b| {
        b.iter(|| {
            black_box(
                bethe
                    .run(&graph)
                    .expect("Bethe sprinkler_network run failed"),
            );
        });
    });

    // Grid MRF
    for (rows, cols) in [(2, 2), (3, 3)] {
        let grid = create_grid_mrf(rows, cols, 2);

        let num_vars = rows * cols;
        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", rows, cols)),
            &grid,
            |b, g| {
                b.iter(|| {
                    black_box(bethe.run(g).expect("Bethe grid_size run failed"));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Tree-Reweighted BP
fn bench_trw_bp(c: &mut Criterion) {
    let mut group = c.benchmark_group("trw_bp");

    let graph = create_sprinkler_network();
    let mut trw = TreeReweightedBP::new(50, 1e-4);
    trw.initialize_uniform_weights(&graph);

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network", |b| {
        b.iter(|| {
            black_box(trw.run(&graph).expect("TRW sprinkler_network run failed"));
        });
    });

    // Grid MRF
    for (rows, cols) in [(2, 2), (3, 3)] {
        let grid = create_grid_mrf(rows, cols, 2);
        let mut trw_grid = TreeReweightedBP::new(20, 1e-4);
        trw_grid.initialize_uniform_weights(&grid);

        let num_vars = rows * cols;
        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", rows, cols)),
            &grid,
            |b, g| {
                b.iter(|| {
                    black_box(trw_grid.run(g).expect("TRW grid_size run failed"));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Expectation Propagation
fn bench_expectation_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation_propagation");

    let graph = create_sprinkler_network();
    let ep = ExpectationPropagation::new(50, 1e-4, 0.0);

    group.throughput(Throughput::Elements(3));
    group.bench_function("sprinkler_network", |b| {
        b.iter(|| {
            black_box(ep.run(&graph).expect("EP sprinkler_network run failed"));
        });
    });

    // Chain MRF
    for length in [5, 10] {
        let chain = create_chain_mrf(length, 2);

        group.throughput(Throughput::Elements(length as u64));
        group.bench_with_input(BenchmarkId::new("chain_length", length), &chain, |b, g| {
            b.iter(|| {
                black_box(ep.run(g).expect("EP chain_length run failed"));
            });
        });
    }

    group.finish();
}

/// Benchmark Gibbs sampling
fn bench_gibbs_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gibbs_sampling");

    let graph = create_sprinkler_network();

    for num_samples in [100, 500, 1000] {
        let sampler = GibbsSampler::new(50, num_samples, 5);

        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("num_samples", num_samples),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(sampler.run(g).expect("Gibbs num_samples run failed"));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark algorithm comparison on the same graph
fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");

    let graph = create_chain_mrf(8, 2);

    // Variable Elimination
    group.bench_function("ve_chain8", |b| {
        b.iter(|| {
            let ve = VariableElimination::new();
            black_box(
                ve.marginalize(&graph, "X_0")
                    .expect("VE ve_chain8 marginalize failed"),
            );
        });
    });

    // Junction Tree
    group.bench_function("jt_chain8", |b| {
        b.iter(|| {
            let mut tree = JunctionTree::from_factor_graph(&graph)
                .expect("JT jt_chain8: from_factor_graph failed");
            tree.calibrate().expect("JT jt_chain8: calibrate failed");
            black_box(
                tree.query_marginal("X_0")
                    .expect("JT jt_chain8: query_marginal failed"),
            );
        });
    });

    // Sum-Product BP
    group.bench_function("bp_chain8", |b| {
        let bp = SumProductAlgorithm::default();
        b.iter(|| {
            black_box(bp.run(&graph).expect("BP bp_chain8 run failed"));
        });
    });

    // Mean-Field
    group.bench_function("mf_chain8", |b| {
        let mf = MeanFieldInference::new(100, 1e-4);
        b.iter(|| {
            black_box(mf.run(&graph).expect("MF mf_chain8 run failed"));
        });
    });

    // Bethe
    group.bench_function("bethe_chain8", |b| {
        let bethe = BetheApproximation::new(50, 1e-4, 0.0);
        b.iter(|| {
            black_box(bethe.run(&graph).expect("Bethe bethe_chain8 run failed"));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_variable_elimination,
    bench_junction_tree,
    bench_belief_propagation,
    bench_mean_field,
    bench_bethe,
    bench_trw_bp,
    bench_expectation_propagation,
    bench_gibbs_sampling,
    bench_algorithm_comparison,
);
criterion_main!(benches);
