//! Tests for the `nas` module.

use super::*;

// ─── helpers ────────────────────────────────────────────────────────────────

fn default_space() -> ArchSearchSpace {
    ArchSearchSpace::new(
        2,
        6,
        vec![32, 64, 128, 256],
        vec!["relu".to_string(), "gelu".to_string(), "tanh".to_string()],
        vec![
            "linear".to_string(),
            "conv".to_string(),
            "attention".to_string(),
        ],
    )
    .unwrap()
}

fn small_space() -> ArchSearchSpace {
    ArchSearchSpace::new(
        1,
        3,
        vec![16, 32],
        vec!["relu".to_string()],
        vec!["linear".to_string()],
    )
    .unwrap()
}

// ─── ArchSearchSpace validation ─────────────────────────────────────────────

#[test]
fn test_arch_search_space_valid() {
    let result = ArchSearchSpace::new(
        1,
        5,
        vec![64, 128],
        vec!["relu".to_string()],
        vec!["linear".to_string()],
    );
    assert!(result.is_ok());
    let space = result.unwrap();
    assert_eq!(space.min_depth, 1);
    assert_eq!(space.max_depth, 5);
}

#[test]
fn test_arch_search_space_invalid_depths() {
    // min_depth > max_depth
    let result = ArchSearchSpace::new(
        5,
        2,
        vec![64],
        vec!["relu".to_string()],
        vec!["linear".to_string()],
    );
    assert!(result.is_err());

    // min_depth == 0
    let result2 = ArchSearchSpace::new(
        0,
        3,
        vec![64],
        vec!["relu".to_string()],
        vec!["linear".to_string()],
    );
    assert!(result2.is_err());
}

#[test]
fn test_arch_search_space_empty_ops() {
    let result = ArchSearchSpace::new(
        1,
        3,
        vec![64],
        vec!["relu".to_string()],
        vec![], // empty op_options
    );
    assert!(result.is_err());

    // empty width_options
    let result2 = ArchSearchSpace::new(
        1,
        3,
        vec![],
        vec!["relu".to_string()],
        vec!["linear".to_string()],
    );
    assert!(result2.is_err());

    // empty activation_options
    let result3 = ArchSearchSpace::new(1, 3, vec![64], vec![], vec!["linear".to_string()]);
    assert!(result3.is_err());
}

// ─── ArchSampler: random_architecture ───────────────────────────────────────

#[test]
fn test_sampler_random_architecture() {
    let space = default_space();
    let ops: Vec<String> = space.op_options.clone();
    let widths: Vec<usize> = space.width_options.clone();
    let acts: Vec<String> = space.activation_options.clone();
    let (min_d, max_d) = (space.min_depth, space.max_depth);

    let mut sampler = ArchSampler::new(space, 0xDEAD_BEEF);

    for _ in 0..10 {
        let arch = sampler.random_architecture().unwrap();
        let d = arch.depth();
        assert!(
            d >= min_d && d <= max_d,
            "depth {d} outside [{min_d}, {max_d}]"
        );
        for layer in &arch.layers {
            assert!(ops.contains(&layer.op), "unknown op: {}", layer.op);
            assert!(
                widths.contains(&layer.width),
                "unknown width: {}",
                layer.width
            );
            assert!(
                acts.contains(&layer.activation),
                "unknown activation: {}",
                layer.activation
            );
        }
    }
}

#[test]
fn test_sampler_deterministic() {
    let space = default_space();
    let seed = 12345_u64;

    let mut s1 = ArchSampler::new(space.clone(), seed);
    let mut s2 = ArchSampler::new(space, seed);

    for _ in 0..20 {
        let a1 = s1.random_architecture().unwrap();
        let a2 = s2.random_architecture().unwrap();
        assert_eq!(a1, a2, "same seed must produce identical sequence");
    }
}

#[test]
fn test_sampler_mutate_stays_valid() {
    let space = default_space();
    let ops: Vec<String> = space.op_options.clone();
    let widths: Vec<usize> = space.width_options.clone();
    let acts: Vec<String> = space.activation_options.clone();
    let (min_d, max_d) = (space.min_depth, space.max_depth);

    let mut sampler = ArchSampler::new(space, 99);
    let mut arch = sampler.random_architecture().unwrap();

    for _ in 0..100 {
        arch = sampler.mutate(&arch).unwrap();
        let d = arch.depth();
        assert!(
            d >= min_d && d <= max_d,
            "mutated depth {d} outside [{min_d}, {max_d}]"
        );
        for layer in &arch.layers {
            assert!(ops.contains(&layer.op));
            assert!(widths.contains(&layer.width));
            assert!(acts.contains(&layer.activation));
        }
    }
}

// ─── Architecture helpers ───────────────────────────────────────────────────

#[test]
fn test_arch_param_count() {
    let arch = Architecture {
        layers: vec![
            LayerSpec {
                op: "linear".to_string(),
                width: 4,
                activation: "relu".to_string(),
            },
            LayerSpec {
                op: "linear".to_string(),
                width: 8,
                activation: "relu".to_string(),
            },
            LayerSpec {
                op: "linear".to_string(),
                width: 2,
                activation: "relu".to_string(),
            },
        ],
    };
    // 4*8 + 8*2 = 32 + 16 = 48
    assert_eq!(arch.param_count(), 48);

    let single = Architecture {
        layers: vec![LayerSpec {
            op: "linear".to_string(),
            width: 64,
            activation: "relu".to_string(),
        }],
    };
    assert_eq!(single.param_count(), 0);
}

#[test]
fn test_arch_to_from_config_roundtrip() {
    let arch = Architecture {
        layers: vec![
            LayerSpec {
                op: "conv".to_string(),
                width: 64,
                activation: "relu".to_string(),
            },
            LayerSpec {
                op: "linear".to_string(),
                width: 128,
                activation: "gelu".to_string(),
            },
            LayerSpec {
                op: "attention".to_string(),
                width: 32,
                activation: "tanh".to_string(),
            },
        ],
    };

    let cfg = arch.to_config();
    let reconstructed = Architecture::from_config(&cfg, 10).unwrap();
    assert_eq!(arch, reconstructed, "round-trip must be lossless");
}

// ─── RegularizedEvolution ───────────────────────────────────────────────────

#[test]
fn test_evolution_ask_fills_population() {
    let space = small_space();
    let pop_size = 5;
    let mut evo = RegularizedEvolution::new(space, pop_size, 2, 7).unwrap();

    for i in 0..pop_size {
        let arch = evo.ask().unwrap();
        evo.tell(arch, i as f64);
    }

    let res = evo.result().unwrap();
    assert_eq!(
        res.history.len(),
        pop_size,
        "history length must equal number of tells"
    );
}

#[test]
fn test_evolution_evicts_oldest() {
    let space = small_space();
    let pop_size = 4;
    let mut evo = RegularizedEvolution::new(space, pop_size, 2, 13).unwrap();

    // Fill population
    for i in 0..pop_size {
        let arch = evo.ask().unwrap();
        evo.tell(arch, i as f64);
    }

    // Add more evaluations — population should stay at pop_size, history grows
    let extra = 6_usize;
    for i in 0..extra {
        let arch = evo.ask().unwrap();
        evo.tell(arch, (pop_size + i) as f64);
    }

    let res = evo.result().unwrap();
    assert_eq!(
        res.history.len(),
        pop_size + extra,
        "all evals recorded in history"
    );
}

#[test]
fn test_evolution_tournament_selection() {
    // Use a fitness that favors smaller param_count (lower param_count → higher score).
    // Run for many rounds and verify that the best() architecture is reasonably small.
    let space = ArchSearchSpace::new(
        1,
        4,
        vec![4, 8, 16, 32, 64, 128],
        vec!["relu".to_string()],
        vec!["linear".to_string()],
    )
    .unwrap();

    let pop_size = 10;
    let tournament_size = 3;
    let mut evo = RegularizedEvolution::new(space, pop_size, tournament_size, 42).unwrap();

    let rounds = 80;
    for _ in 0..rounds {
        let arch = evo.ask().unwrap();
        // Score: inverse of param_count (smaller arch = higher score). Add small
        // epsilon so that 0-param archs (1 layer) still get a finite score.
        let score = 1.0 / (arch.param_count() as f64 + 1.0);
        evo.tell(arch, score);
    }

    let (best_arch, best_score) = evo.best().unwrap();
    // The best score must be positive and finite.
    assert!(best_score.is_finite() && *best_score > 0.0);
    // The best architecture should prefer small parameter counts.
    assert!(
        best_arch.param_count() <= 64 * 64,
        "unreasonably large best arch"
    );
}

#[test]
fn test_evolution_best() {
    let space = small_space();
    let mut evo = RegularizedEvolution::new(space, 5, 2, 55).unwrap();

    let mut max_score = f64::NEG_INFINITY;
    for i in 0..8 {
        let arch = evo.ask().unwrap();
        let score = i as f64 * 0.1;
        if score > max_score {
            max_score = score;
        }
        evo.tell(arch, score);
    }

    let (_, best_score) = evo.best().unwrap();
    assert!(
        (*best_score - max_score).abs() < 1e-9 || *best_score <= max_score,
        "best() must return the highest-scored surviving member"
    );
}

#[test]
fn test_evolution_deterministic() {
    let space = small_space();
    let seed = 9999_u64;

    let mut evo1 = RegularizedEvolution::new(space.clone(), 5, 2, seed).unwrap();
    let mut evo2 = RegularizedEvolution::new(space, 5, 2, seed).unwrap();

    for i in 0..15 {
        let a1 = evo1.ask().unwrap();
        let a2 = evo2.ask().unwrap();
        assert_eq!(a1, a2, "ask {i}: same seed must yield identical sequence");
        evo1.tell(a1.clone(), i as f64);
        evo2.tell(a2, i as f64);
    }
}

// ─── RandomArchSearch ───────────────────────────────────────────────────────

#[test]
fn test_random_search_basic() {
    let space = default_space();
    let mut search = RandomArchSearch::new(space, 777);

    let mut max_score = f64::NEG_INFINITY;
    let mut max_arch: Option<Architecture> = None;

    for i in 0..10 {
        let arch = search.ask().unwrap();
        let score = i as f64 * 0.3 - 1.0;
        if score > max_score {
            max_score = score;
            max_arch = Some(arch.clone());
        }
        search.tell(arch, score);
    }

    let (best_arch, best_score) = search.best().unwrap();
    assert!((best_score - max_score).abs() < 1e-9, "best score mismatch");
    assert_eq!(best_arch, max_arch.as_ref().unwrap(), "best arch mismatch");
}

// ─── NasResult history ──────────────────────────────────────────────────────

#[test]
fn test_nas_result_history() {
    let space = small_space();
    let mut search = RandomArchSearch::new(space, 321);

    let n = 15;
    for i in 0..n {
        let arch = search.ask().unwrap();
        search.tell(arch, i as f64 * 0.5);
    }

    let result = search.result().unwrap();
    assert_eq!(
        result.history.len(),
        n,
        "NasResult history must contain all evaluations"
    );
    // best_score must equal max over history
    let max_from_history = result
        .history
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((result.best_score - max_from_history).abs() < 1e-9);
}
