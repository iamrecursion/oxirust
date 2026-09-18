# oxiphysics-articulated

Featherstone articulated-body dynamics (RNEA + ABA) for OxiPhysics.

## Features

- **RNEA** — Recursive Newton-Euler Algorithm: inverse dynamics, computes joint torques from `(q, q̇, q̈)`
- **ABA** — Articulated Body Algorithm: forward dynamics, computes `q̈` from `(q, q̇, τ)`
- **CRBA** — Composite Rigid Body Algorithm: O(n²) joint-space mass matrix
- **OSC** — Operational Space Control: inertia `Λ = (J·M⁻¹·Jᵀ)⁻¹`
- **Centroidal dynamics** — Centroidal Momentum Matrix (CMM)
- **Spatial math** — 6D spatial vectors, spatial inertia, Plücker transforms
- **Joint types** — Revolute, Prismatic, Fixed, FreeFloating, Universal, Spherical, Helical
- **Joint limits** — Soft joint limits via penalty forces (`JointLimit`, `JointLimitSet`)
- **Derivatives** — Analytic RNEA/ABA derivatives
- **IK** — Damped least-squares inverse kinematics with null-space posture task
- **Sparse LᵀDL** — Branch-induced-sparsity mass-matrix factorization

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
oxiphysics-articulated = "0.1.3"
```

Construct a model and run inverse or forward dynamics. Bodies are added root
first; `add_body` takes the body (inertia, parent index, fixed parent-to-joint
transform) together with the boxed joint connecting it to its parent:

```rust
use oxiphysics_articulated::{
    aba::aba,
    body::RigidBody,
    joint::RevoluteJoint,
    model::ArticulatedModel,
    rnea::rnea,
    spatial::{SpatialInertia, SpatialTransform},
};

// Gravity along -y.
let mut model = ArticulatedModel::new([0.0, -9.81, 0.0]);
let link = SpatialInertia::from_com(
    1.0,
    [0.5, 0.0, 0.0],
    [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
);

// Two-link planar arm: both joints rotate about z.
let link0 = model.add_body(
    RigidBody::new("link0", link, None, SpatialTransform::IDENTITY),
    Box::new(RevoluteJoint::new([0.0, 0.0, 1.0])),
);
model.add_body(
    RigidBody::new("link1", link, Some(link0), SpatialTransform::from_translation([1.0, 0.0, 0.0])),
    Box::new(RevoluteJoint::new([0.0, 0.0, 1.0])),
);

let q = [0.3, -0.2];   // joint positions
let qd = [0.1, 0.4];   // joint velocities
let qdd = [0.5, -1.0]; // joint accelerations

// Inverse dynamics: given motion -> torques
let tau = rnea(&model, &q, &qd, &qdd);

// Forward dynamics: given torques -> accelerations (recovers qdd)
let qdd_out = aba(&model, &q, &qd, &tau);
```

Fallible variants report bad input as `ArticulatedError` instead of panicking:
`ArticulatedModel::try_add_body`, `RevoluteJoint::try_new` (and the other
joints' `try_new`), `aba::try_aba`, and `aba::aba_with_workspace`, which reuses
an `AbaWorkspace` across simulation steps so forward dynamics does not allocate.

Further modules: `rnea_derivatives` / `aba_derivatives` (analytic dynamics
derivatives), `ik` (forward kinematics, geometric Jacobian, damped
least-squares IK with optional singularity-aware damping) and `ltdl` (sparse
`LᵀDL` mass-matrix factorization).

### URDF import

`oxiphysics_io::robotics_io::urdf_to_articulated` (crate `oxiphysics-io`)
builds an `ArticulatedModel` from a parsed URDF robot.

## Part of OxiPhysics

This crate is part of the [OxiPhysics](https://github.com/cool-japan/oxiphysics) ecosystem — a pure-Rust physics engine built on Featherstone's rigid-body dynamics algorithms.

## License

Apache-2.0
