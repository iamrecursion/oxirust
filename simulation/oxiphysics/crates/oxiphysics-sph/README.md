# oxiphysics-sph

**Status: Stable** | Version 0.1.3 | Part of [OxiPhysics](https://github.com/cool-japan/oxiphysics)

Smoothed-Particle Hydrodynamics (SPH) simulation for the OxiPhysics engine — Pure Rust, no C/Fortran dependencies.

## Domain Highlights

- **Pressure solvers**: IISPH, PCISPH, DFSPH (full divergence-free variant), WCSPH
- **Multiphase & free-surface**: immiscible fluids, free-surface tracking, open boundaries
- **Surface tension**: Continuum Surface Force (CSF) model
- **Turbulence**: SPH turbulence models with sub-grid scale closures
- **Adaptive resolution**: adaptive smoothing length (`adaptive_h`), adaptive SPH kernels
- **Granular flow**: granular SPH with yield-stress rheology
- **Coupling**: SPH↔rigid, SPH↔FEM, SPH↔DEM coupling layers
- **Timestep control**: CFL-based adaptive timestepping

## Publicly Exported Modules (27)

`adaptive`, `adaptive_h`, `adaptive_sph`, `boundary_sph`, `coupling`, `dfsph`, `dfsph_full`,
`free_surface`, `granular`, `iisph`, `immiscible`, `kernel`, `multiphase`, `neighbor`,
`open_boundary`, `particle`, `pcisph`, `pressure_solvers`, `simulation`, `surface`,
`surface_tension`, `timestep`, `timestepping`, `turbulence`, `turbulence_sph`, `viscosity`, `wcsph`

## Key APIs

```rust
use oxiphysics_sph::{SphSolver, IisphSolver, PcisphSolver, CsfSurfaceTension};
use oxiphysics_sph::{adaptive_sph::*, free_surface::*, multiphase::*, turbulence_sph::*};
```

- `SphSolver` trait — common interface across all pressure-solver backends
- `IisphSolver` — Implicit Incompressible SPH
- `PcisphSolver` — Predictive-Corrective Incompressible SPH
- `CsfSurfaceTension` — Continuum Surface Force surface-tension model

## Cargo features

- `parallel` (off by default) — evaluates the per-particle gather loops of the
  weakly-compressible SPH pipeline (density summation, pressure / Morris viscosity /
  continuum-surface-force accelerations of `WcSphSim`, and the `wcsph` functions for
  density, pressure, viscosity, XSPH, tensile, Shepard/MLS, Riemann, particle-shifting and
  δ-SPH diffusion terms) on the [rayon](https://crates.io/crates/rayon) thread pool. Every
  particle still sums its own neighbours in list order, so the results are bitwise
  identical to the serial loops for every thread count. Pure Rust; benchmark with
  `cargo bench -p oxiphysics-sph --features parallel --bench wcsph_parallel`.

## Statistics

| Metric | Count |
|---|---|
| Public items | 5,066 |
| Tests | 4,370 |
| Stubs | 0 |

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team KitaSan)

