import json, sys

verification_notes = sys.argv[1] if len(sys.argv) > 1 else ""

report = {
  "package_id": "w2-core-05",
  "completed": ["F0599","F0600","F0602","F0603","F0605","F0606","F0609","F0612","F0616","F0620"],
  "skipped": [],
  "files_changed": [
    "crates/oxiphysics-core/src/error.rs",
    "crates/oxiphysics-core/src/differential_geometry/types.rs",
    "crates/oxiphysics-core/src/differential_geometry/functions.rs",
    "crates/oxiphysics-core/src/differential_geometry/mod.rs",
    "crates/oxiphysics-core/src/differential_geometry/metric_regression_tests.rs",
    "crates/oxiphysics-core/src/dynamic_systems.rs",
    "crates/oxiphysics-core/src/dynamic_systems/tests.rs",
    "crates/oxiphysics-core/src/dynamic_systems/regression_tests.rs",
    "crates/oxiphysics-core/src/dynamical_systems.rs",
    "crates/oxiphysics-core/src/dynamical_systems/tests.rs",
    "crates/oxiphysics-core/src/dynamical_systems/lyapunov_tests.rs",
    "crates/oxiphysics-core/src/ergodic_theory.rs",
    "crates/oxiphysics-core/src/ergodic_theory/rqa_tests.rs"
  ],
  "tests_added": 70,
  "summary": (
    "All 10 findings fixed in oxiphysics-core (F0605 and F0606 describe the same instantaneous_phase fix). "
    "Singular metrics now return PhysicsError (scaled-pivot MetricTensorND::try_inverse; the N-dimensional differential-geometry API became Result-returning, no callers outside the crate), "
    "christoffel_111 is corrected, the correlation sum is streamed in O(n_radii) memory, Rosenstein has a Theiler window (plus auto/default helpers), instantaneous_phase uses the FFT analytic signal, "
    "the Lyapunov routines advance tangents with RK4 stage Jacobians (the new tests pin the Lorenz lambda1 and the maximal exponent to 0.906 +- 0.03 and the exponent sum to -(sigma+1+beta) +- 0.02), RQA DET honours min_line and excludes the line of identity, and check_md_timestep/check_lbm_mach reject NaN and negative inputs. "
    "The inline test modules of dynamical_systems.rs (1998 lines) and dynamic_systems.rs were relocated to <module>/tests.rs to respect the 2000-line limit."
  ),
  "changelog": [
    "`oxiphysics-core`: Added `MetricTensorND::try_inverse`, which returns `PhysicsError::InvalidInput` (field `metric` or `dim`) for singular, non-finite or oversized metrics using a pivot test relative to each row's scale, so SI-unit and badly scaled metrics stay invertible; `MetricTensorND::inverse` now documents its panics and delegates to it, and `determinant` returns 1 for `dim == 0` and NaN for `dim > MAX_DIM` instead of panicking.",
    "`oxiphysics-core`: BREAKING: `LeviCivitaConnection::from_metric_fn`, `RiemannTensor::from_metric_fn`, `RicciTensor::from_metric_fn`, `ricci_scalar`, `ricci_scalar_from_metric_fn`, `einstein_tensor`, `kretschner_scalar`, `weyl_tensor`, `geodesic_equation_solve`, `parallel_transport_nd`, `killing_equation_violation`, `geodesic_deviation`, `gauss_bonnet_2d_integrand` and `MetricTensorND::raise_index` now return `Result<_, PhysicsError>` and report singular metrics (for example polar coordinates at `r = 0`), `dim > MAX_DIM`, a zero or non-finite finite-difference step and (Weyl) `dim < 3` as errors instead of panicking; new `GeodesicPoint` alias.",
    "`oxiphysics-core`: Fixed `RiemannianMetric::christoffel_111` (wrong `F_u` and wrong sign of the `g^12` term).",
    "`oxiphysics-core`: `AttractorAnalysis::correlation_dimension` bins squared distances in two streaming passes (O(`n_radii`) memory, no sort, no square roots) instead of storing all n^2/2 distances, takes `d_min` from the smallest positive distance (coincident points no longer give `ln 0`) and ignores NaN pairs; added `correlation_dimension_theiler`.",
    "`oxiphysics-core`: BREAKING: `AttractorAnalysis::rosenstein_lyapunov(horizon)` is now `rosenstein_lyapunov(horizon, theiler_window)` so the nearest neighbour is no longer the point's own orbit neighbour; added `rosenstein_lyapunov_auto` and `default_theiler_window` (first autocorrelation zero crossing of coordinate 0), a squared-distance neighbour search, a full-horizon ensemble and NaN (instead of 0) where no pair can be evaluated.",
    "`oxiphysics-core`: `SynchronizationAnalysis::instantaneous_phase` now uses the FFT analytic signal (Hilbert transform) and returns a truly unwrapped phase instead of a cumulative-sum pseudo-quadrature; constant signals return zeros.",
    "`oxiphysics-core`: `lyapunov_exponents_qr` and `maximal_lyapunov_exponent` advance the tangent vectors with RK4 using the Jacobian of every stage (new `rk4_step_with_tangents`) instead of Euler with the Jacobian at the end of the step (tests: Lorenz lambda1 and maximal exponent within 0.906 +- 0.03, exponent sum within -(sigma+1+beta) +- 0.02, harmonic oscillator sum below 1e-6); `maximal_lyapunov_exponent` treats `renorm == 0` as 1 instead of panicking and `kaplan_yorke_dim(&[])` returns 0 instead of panicking.",
    "`oxiphysics-core`: `RecurrenceAnalysis::compute` honours `min_line` for DET, and the main diagonal (line of identity) is excluded from DET, `L_mean`, `L_max` and the diagonal entropy (`max_diagonal_length` was n for every signal); added `determinism_with_min_line`.",
    "`oxiphysics-core`: `check_md_timestep` rejects NaN, infinite and non-positive time steps and bounds, and `check_lbm_mach` uses `|velocity| / cs` and rejects non-finite or non-positive `cs` and non-finite Mach numbers."
  ],
  "verification": {"check": True, "clippy": True, "tests": "passed", "notes": verification_notes}
}
path = "/Users/kitasan/.claude/projects/-Users-kitasan-work-oxiphysics/campaign/wave_results/w2/w2-core-05.report.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False))
