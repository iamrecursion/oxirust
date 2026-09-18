//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{PipelineConfig, PipelineError, SphereContact, StateQuantity, WorldState};
use crate::gpu_radix::radix_sort_pairs_gpu;

/// Magnitude of the gravitational acceleration applied by [`run_integration`].
const STANDARD_GRAVITY: f64 = 9.81;
/// Gravitational acceleration applied to every dynamic body.
const GRAVITY: [f64; 3] = [0.0, -STANDARD_GRAVITY, 0.0];
/// Depth, as a fraction of the smaller sphere radius, up to which a penetration is
/// tolerated and not corrected (Baumgarte slop).  It keeps resting contacts from
/// switching on and off.
const PENETRATION_SLOP_FRACTION: f64 = 0.01;
/// Largest penetration, as a fraction of the smaller sphere radius, that the
/// Baumgarte term tries to remove in one step.  It bounds the push-out speed of
/// deeply overlapping spheres.
const MAX_CORRECTION_FRACTION: f64 = 0.2;
/// A contact that closes slower than this many time steps of gravity is a resting
/// contact and does not bounce.
const RESTING_STEPS_OF_GRAVITY: f64 = 2.0;
/// Centres closer than this fraction of the radius sum have no usable direction.
const COINCIDENT_CENTER_FRACTION: f64 = 1e-12;
/// Contact normal for coincident centres.
const FALLBACK_NORMAL: [f64; 3] = [1.0, 0.0, 0.0];

/// Returns `true` for an inverse mass that lets a body move (finite and positive).
fn is_movable(inverse_mass: f64) -> bool {
    inverse_mass.is_finite() && inverse_mass > 0.0
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(v: [f64; 3], factor: f64) -> [f64; 3] {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

/// A collidable body of the broad phase: the bounding box of its sphere.
#[derive(Debug, Clone, Copy)]
struct SweepEntry {
    body: usize,
    lo: [f64; 3],
    hi: [f64; 3],
    dynamic: bool,
}
/// Buffers the broad phase reuses from one sub-step to the next.
#[derive(Debug, Default)]
pub(super) struct BroadphaseScratch {
    entries: Vec<SweepEntry>,
    order: Vec<usize>,
}

/// The axis along which the centres of `entries` spread the most (lowest axis on a
/// tie).  Sweeping along it keeps the active set of the sweep small: a vertical
/// pile of spheres is swept along `y`, where its bodies are far apart, instead of
/// along `x`, where all of them overlap.
fn widest_axis(entries: &[SweepEntry]) -> usize {
    let count = entries.len() as f64;
    let mut best_axis = 0;
    let mut best_variance = -1.0;
    for axis in 0..3 {
        let centre = |entry: &SweepEntry| 0.5 * (entry.lo[axis] + entry.hi[axis]);
        let mean = entries.iter().map(centre).sum::<f64>() / count;
        let variance = entries
            .iter()
            .map(|entry| (centre(entry) - mean).powi(2))
            .sum::<f64>()
            / count;
        if variance > best_variance {
            best_variance = variance;
            best_axis = axis;
        }
    }
    best_axis
}

/// Map `x` to a `u64` whose unsigned order is the IEEE total order of `x`
/// ([`f64::total_cmp`]).
fn ordered_key(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

/// Entry indices sorted by the lower bound along `axis` (ties: lower index first),
/// computed by the GPU radix sort.
///
/// The 64-bit order key of the bound is sorted in two stable 32-bit passes (least
/// significant word first), so the result is exactly the order of the CPU sort in
/// [`sort_entries`].  `None` if the entries cannot be indexed by `u32` or the sort
/// returned an unexpected number of entries.
fn gpu_sorted_order(entries: &[SweepEntry], axis: usize) -> Option<Vec<usize>> {
    let ids = (0..entries.len())
        .map(|index| u32::try_from(index).ok())
        .collect::<Option<Vec<u32>>>()?;
    let keys: Vec<u64> = entries
        .iter()
        .map(|entry| ordered_key(entry.lo[axis]))
        .collect();
    let low_words: Vec<u32> = keys.iter().map(|key| (key & 0xFFFF_FFFF) as u32).collect();
    let (_, by_low_word) = radix_sort_pairs_gpu(&low_words, &ids);
    let high_words: Vec<u32> = by_low_word
        .iter()
        .map(|&id| {
            keys.get(id as usize)
                .map_or(u32::MAX, |key| (key >> 32) as u32)
        })
        .collect();
    let (_, order) = radix_sort_pairs_gpu(&high_words, &by_low_word);
    if order.len() != entries.len() {
        return None;
    }
    Some(order.into_iter().map(|id| id as usize).collect())
}

/// Fill `scratch.order` with the entry indices sorted by the lower bound along
/// `axis`, ties broken by the lower index (a total order, so the result does not
/// depend on the sorting algorithm).  `f64::total_cmp` keeps the sort total for
/// every input, including infinite bounds.
fn sort_entries(scratch: &mut BroadphaseScratch, axis: usize, use_gpu: bool) {
    if use_gpu && let Some(order) = gpu_sorted_order(&scratch.entries, axis) {
        scratch.order = order;
        return;
    }
    let entries = &scratch.entries;
    scratch.order.clear();
    scratch.order.extend(0..entries.len());
    scratch.order.sort_unstable_by(|&first, &second| {
        entries[first].lo[axis]
            .total_cmp(&entries[second].lo[axis])
            .then(first.cmp(&second))
    });
}

/// Returns `true` if the boxes of `a` and `b` overlap on every axis (touching counts).
fn boxes_overlap(a: &SweepEntry, b: &SweepEntry) -> bool {
    (0..3).all(|axis| a.lo[axis] <= b.hi[axis] && b.lo[axis] <= a.hi[axis])
}

/// Sweep the sorted entries and push the pairs whose boxes overlap on all axes.
///
/// For an entry `a`, every later entry `b` in the order has `b.lo >= a.lo` along
/// the sweep axis, so the intervals overlap exactly while `b.lo <= a.hi`; the first
/// later entry that starts behind `a.hi` ends the inner loop.  Pairs of two
/// immovable bodies can never produce a constraint and are skipped.
fn sweep(entries: &[SweepEntry], order: &[usize], axis: usize, pairs: &mut Vec<(usize, usize)>) {
    for (position, &first) in order.iter().enumerate() {
        let Some(a) = entries.get(first) else {
            continue;
        };
        for &second in order.iter().skip(position + 1) {
            let Some(b) = entries.get(second) else {
                continue;
            };
            if b.lo[axis] > a.hi[axis] {
                break;
            }
            if (a.dynamic || b.dynamic) && boxes_overlap(a, b) {
                pairs.push((a.body.min(b.body), a.body.max(b.body)));
            }
        }
    }
}

/// Broad phase: sort-and-sweep over the bounding boxes of the body spheres.
///
/// Fills `pairs` with every pair `(a, b)`, `a < b`, of collidable bodies whose
/// boxes overlap (touching counts) and that are not both immovable, in ascending
/// order.  Bodies without a collision shape or with a non-finite position take no
/// part.  With `use_gpu` the sort runs on the GPU radix sort; the pairs are the
/// same either way.
pub(super) fn run_broadphase(
    world: &WorldState,
    use_gpu: bool,
    scratch: &mut BroadphaseScratch,
    pairs: &mut Vec<(usize, usize)>,
) {
    pairs.clear();
    scratch.entries.clear();
    for body in 0..world.body_count() {
        let (Some(centre), Some(radius)) = (world.position(body), world.collision_radius(body))
        else {
            continue;
        };
        scratch.entries.push(SweepEntry {
            body,
            lo: centre.map(|c| c - radius),
            hi: centre.map(|c| c + radius),
            dynamic: world.is_dynamic(body),
        });
    }
    if scratch.entries.len() < 2 {
        return;
    }
    let axis = widest_axis(&scratch.entries);
    sort_entries(scratch, axis, use_gpu);
    sweep(&scratch.entries, &scratch.order, axis, pairs);
    pairs.sort_unstable();
}

/// Sphere-sphere contact of the bodies `a` and `b`, if their spheres penetrate.
fn sphere_contact(world: &WorldState, a: usize, b: usize) -> Option<SphereContact> {
    let (centre_a, radius_a) = (world.position(a)?, world.collision_radius(a)?);
    let (centre_b, radius_b) = (world.position(b)?, world.collision_radius(b)?);
    let offset = sub(centre_b, centre_a);
    let reach = radius_a + radius_b;
    let distance_squared = dot(offset, offset);
    if distance_squared >= reach * reach {
        return None;
    }
    let distance = distance_squared.sqrt();
    let depth = reach - distance;
    if depth <= 0.0 {
        return None;
    }
    let normal = if distance > COINCIDENT_CENTER_FRACTION * reach {
        scale(offset, 1.0 / distance)
    } else {
        FALLBACK_NORMAL
    };
    let along = radius_a - 0.5 * depth;
    let point = [
        centre_a[0] + normal[0] * along,
        centre_a[1] + normal[1] * along,
        centre_a[2] + normal[2] * along,
    ];
    Some(SphereContact {
        body_a: a,
        body_b: b,
        normal,
        point,
        depth,
        normal_impulse: 0.0,
    })
}

/// Narrow phase: replace `contacts` with the contact of every broad-phase pair
/// whose spheres really penetrate (sphere-sphere normal and depth).
pub(super) fn run_narrowphase(
    world: &WorldState,
    pairs: &[(usize, usize)],
    contacts: &mut Vec<SphereContact>,
) {
    contacts.clear();
    contacts.extend(
        pairs
            .iter()
            .filter_map(|&(a, b)| sphere_contact(world, a, b)),
    );
}

/// Parameters of the contact solver, already sanitised.
#[derive(Debug, Clone, Copy)]
pub(super) struct SolverSettings {
    iterations: u32,
    restitution: f64,
    baumgarte: f64,
}
impl SolverSettings {
    /// Read the solver parameters of `config`: at least one iteration, restitution
    /// and Baumgarte factor clamped to `[0, 1]` (`NaN` counts as `0`).
    pub(super) fn from_config(config: &PipelineConfig) -> Self {
        let unit = |value: f64| {
            if value.is_nan() {
                0.0
            } else {
                value.clamp(0.0, 1.0)
            }
        };
        Self {
            iterations: config.solver_iterations.max(1),
            restitution: unit(config.restitution),
            baumgarte: unit(config.baumgarte),
        }
    }
}
/// Per-contact constants of the velocity solve.
#[derive(Debug, Clone, Copy)]
struct SolverRow {
    contact: usize,
    inverse_mass_a: f64,
    inverse_mass_b: f64,
    inverse_effective_mass: f64,
    target_speed: f64,
}

/// Build the solver row of `contact`, or `None` if it carries no constraint: both
/// bodies immovable, or a velocity that is not finite.
///
/// The row's target is the separating speed along the normal the solver enforces:
/// the larger of the Baumgarte push-out speed
/// `baumgarte / dt * clamp(depth - slop, 0, max_correction)` and the rebound speed
/// `restitution * closing speed` of a contact that closes faster than `resting_speed`.
fn solver_row(
    world: &WorldState,
    index: usize,
    contact: &SphereContact,
    settings: SolverSettings,
    dt: f64,
    resting_speed: f64,
) -> Option<SolverRow> {
    let inverse_mass_a = world.inverse_mass(contact.body_a);
    let inverse_mass_b = world.inverse_mass(contact.body_b);
    let effective_mass_inverse = inverse_mass_a + inverse_mass_b;
    if effective_mass_inverse <= 0.0 {
        return None;
    }
    let velocity_a = world.velocity(contact.body_a)?;
    let velocity_b = world.velocity(contact.body_b)?;
    let smaller_radius = world
        .collision_radius(contact.body_a)?
        .min(world.collision_radius(contact.body_b)?);
    let normal_speed = dot(sub(velocity_b, velocity_a), contact.normal);
    let push_out_speed = if dt > 0.0 {
        let slop = PENETRATION_SLOP_FRACTION * smaller_radius;
        let largest_correction = MAX_CORRECTION_FRACTION * smaller_radius;
        let correction = (contact.depth - slop).max(0.0).min(largest_correction);
        settings.baumgarte / dt * correction
    } else {
        0.0
    };
    let rebound_speed = if normal_speed < -resting_speed {
        -settings.restitution * normal_speed
    } else {
        0.0
    };
    Some(SolverRow {
        contact: index,
        inverse_mass_a,
        inverse_mass_b,
        inverse_effective_mass: 1.0 / effective_mass_inverse,
        target_speed: push_out_speed.max(rebound_speed),
    })
}

/// Constraint solve: sequential impulses on the non-penetration contacts.
///
/// Every contact asks for a separating speed of at least its target (see
/// [`solver_row`]).  The solver sweeps the contacts `settings.iterations` times
/// (Gauss-Seidel); the impulse of each contact accumulates and is clamped to be
/// non-negative, so contacts can push but never pull.  The accumulated impulses are
/// stored in `contacts`.
///
/// Returns the number of contacts that carry a constraint (at least one movable
/// body and finite velocities).
pub(super) fn run_constraint_solve(
    world: &mut WorldState,
    contacts: &mut [SphereContact],
    settings: SolverSettings,
    dt: f64,
) -> usize {
    let resting_speed = RESTING_STEPS_OF_GRAVITY * STANDARD_GRAVITY * dt;
    let rows: Vec<SolverRow> = contacts
        .iter_mut()
        .enumerate()
        .filter_map(|(index, contact)| {
            contact.normal_impulse = 0.0;
            solver_row(world, index, contact, settings, dt, resting_speed)
        })
        .collect();
    for _ in 0..settings.iterations {
        for row in &rows {
            let Some(contact) = contacts.get_mut(row.contact) else {
                continue;
            };
            let (Some(velocity_a), Some(velocity_b)) = (
                world.velocity(contact.body_a),
                world.velocity(contact.body_b),
            ) else {
                continue;
            };
            let normal_speed = dot(sub(velocity_b, velocity_a), contact.normal);
            let accumulated = (contact.normal_impulse
                + (row.target_speed - normal_speed) * row.inverse_effective_mass)
                .max(0.0);
            let applied = accumulated - contact.normal_impulse;
            contact.normal_impulse = accumulated;
            world.add_to_velocity(
                contact.body_a,
                scale(contact.normal, -applied * row.inverse_mass_a),
            );
            world.add_to_velocity(
                contact.body_b,
                scale(contact.normal, applied * row.inverse_mass_b),
            );
        }
    }
    rows.len()
}

/// Integration: semi-implicit Euler with gravity `[0, -9.81, 0]`.
///
/// Bodies with an inverse mass that is not finite and positive do not move.
pub(super) fn run_integration(world: &mut WorldState, dt: f64) {
    let bodies = world
        .positions
        .chunks_exact_mut(3)
        .zip(world.velocities.chunks_exact_mut(3))
        .zip(&world.inverse_masses);
    for ((position, velocity), &inverse_mass) in bodies {
        if !is_movable(inverse_mass) {
            continue;
        }
        for ((p, v), g) in position.iter_mut().zip(velocity.iter_mut()).zip(GRAVITY) {
            *v += g * dt;
            *p += *v * dt;
        }
    }
}

/// Index of the first body whose three values in the flat vector array `values`
/// are not all finite.
fn first_non_finite_body(values: &[f64]) -> Option<usize> {
    values
        .chunks(3)
        .position(|triple| triple.iter().any(|value| !value.is_finite()))
}

/// Post-process: check that the state the step produced is valid.
///
/// # Errors
///
/// [`PipelineError::NonFiniteState`] for the first body with a non-finite position,
/// velocity, radius or inverse mass (checked in that order), and
/// [`PipelineError::NegativeInverseMass`] for a negative inverse mass.
pub(super) fn run_postprocess(world: &WorldState) -> Result<(), PipelineError> {
    let non_finite = |body: usize, quantity: StateQuantity| {
        Err(PipelineError::NonFiniteState { body, quantity })
    };
    if let Some(body) = first_non_finite_body(&world.positions) {
        return non_finite(body, StateQuantity::Position);
    }
    if let Some(body) = first_non_finite_body(&world.velocities) {
        return non_finite(body, StateQuantity::Velocity);
    }
    if let Some(body) = world.radii.iter().position(|radius| !radius.is_finite()) {
        return non_finite(body, StateQuantity::Radius);
    }
    if let Some(body) = world.inverse_masses.iter().position(|m| !m.is_finite()) {
        return non_finite(body, StateQuantity::InverseMass);
    }
    if let Some((body, &value)) = world
        .inverse_masses
        .iter()
        .enumerate()
        .find(|(_, inverse_mass)| **inverse_mass < 0.0)
    {
        return Err(PipelineError::NegativeInverseMass { body, value });
    }
    Ok(())
}
#[cfg(test)]
mod tests {

    use crate::pipeline::AsyncComputeQueue;

    use crate::pipeline::BarrierSet;
    use crate::pipeline::BufferUsage;

    use crate::pipeline::ComputePipeline;
    use crate::pipeline::CpuBuffer;
    use crate::pipeline::DispatchBatch;

    use crate::pipeline::GpuMemoryPool;

    use crate::pipeline::PhysicsPipeline;
    use crate::pipeline::PipelineBuilder;
    use crate::pipeline::PipelineConfig;
    use crate::pipeline::PipelineProfiler;
    use crate::pipeline::PipelineStage;

    use crate::pipeline::PipelineStats;

    use crate::pipeline::ResourceBarrier;
    use crate::pipeline::ResourceHandle;

    use crate::pipeline::WorldState;
    #[test]
    fn test_pipeline_workgroups() {
        let p = ComputePipeline::new("test", "", "main");
        assert_eq!(p.workgroups_needed(200), [4, 1, 1]);
        assert_eq!(p.workgroups_needed(128), [2, 1, 1]);
        assert_eq!(p.workgroups_needed(1), [1, 1, 1]);
    }
    #[test]
    fn test_cpu_buffer_zeros() {
        let buf = CpuBuffer::new_zeros("zeros", 10, BufferUsage::Storage);
        assert_eq!(buf.len(), 10);
        assert!(buf.data.iter().all(|&v| v == 0.0));
        assert!(!buf.is_empty());
    }
    #[test]
    fn test_dispatch_batch_bind() {
        let pipeline = ComputePipeline::new("batch", "", "main");
        let mut batch = DispatchBatch::new(pipeline, 64);
        batch.bind(CpuBuffer::new_zeros("a", 64, BufferUsage::Storage));
        batch.bind(CpuBuffer::new_zeros("b", 64, BufferUsage::StorageReadOnly));
        batch.bind(CpuBuffer::new_f32("c", vec![1.0; 4], BufferUsage::Uniform));
        assert_eq!(batch.bindings.len(), 3);
    }
    #[test]
    fn default_config_has_all_stages() {
        let cfg = PipelineConfig::new();
        for stage in PipelineStage::all_in_order() {
            assert!(cfg.is_enabled(stage), "stage {stage:?} should be enabled");
        }
        assert_eq!(cfg.substeps, 1);
        assert!(!cfg.use_gpu);
        assert_eq!(
            cfg.solver_iterations,
            crate::pipeline::DEFAULT_SOLVER_ITERATIONS
        );
        assert_eq!(cfg.restitution, 0.0);
        assert_eq!(cfg.baumgarte, crate::pipeline::DEFAULT_BAUMGARTE);
    }
    #[test]
    fn config_disable_stage() {
        let mut cfg = PipelineConfig::new();
        cfg.enabled_stages
            .retain(|&s| s != PipelineStage::PostProcess);
        assert!(!cfg.is_enabled(PipelineStage::PostProcess));
        assert!(cfg.is_enabled(PipelineStage::BroadPhase));
    }
    #[test]
    fn builder_pattern_substeps_and_gpu_flag() {
        let p = PipelineBuilder::new().substeps(4).use_gpu(true).build();
        assert_eq!(p.config.substeps, 4);
        assert!(p.config.use_gpu);
    }
    #[test]
    fn builder_disable_stage() {
        let p = PipelineBuilder::new()
            .disable_stage(PipelineStage::NarrowPhase)
            .build();
        assert!(!p.config.is_enabled(PipelineStage::NarrowPhase));
        assert!(p.config.is_enabled(PipelineStage::BroadPhase));
    }
    #[test]
    fn builder_enable_stage_idempotent() {
        let p = PipelineBuilder::new()
            .enable_stage(PipelineStage::BroadPhase)
            .enable_stage(PipelineStage::BroadPhase)
            .build();
        let count = p
            .config
            .enabled_stages
            .iter()
            .filter(|&&s| s == PipelineStage::BroadPhase)
            .count();
        assert_eq!(count, 1);
    }
    #[test]
    fn stage_order_is_canonical() {
        let order = PipelineStage::all_in_order();
        for w in order.windows(2) {
            assert!(
                w[0] < w[1],
                "stage order violated: {:?} should be < {:?}",
                w[0],
                w[1]
            );
        }
    }
    #[test]
    fn empty_world_step_no_panic() {
        let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
        let mut world = WorldState::default();
        let stats = pipeline
            .step(&mut world, 1.0 / 60.0)
            .expect("an empty world is valid");
        assert_eq!(stats.collision_pairs, 0);
        assert_eq!(stats.contacts, 0);
        assert_eq!(stats.solved_constraints, 0);
    }
    #[test]
    fn stats_accumulate_sums_fields() {
        let mut a = PipelineStats {
            broadphase_ms: 1.0,
            narrowphase_ms: 2.0,
            constraint_ms: 3.0,
            integration_ms: 4.0,
            postprocess_ms: 5.0,
            total_time_ms: 6.0,
            collision_pairs: 10,
            contacts: 6,
            solved_constraints: 5,
        };
        let b = PipelineStats {
            broadphase_ms: 0.5,
            narrowphase_ms: 0.5,
            constraint_ms: 0.5,
            integration_ms: 0.5,
            postprocess_ms: 0.5,
            total_time_ms: 1.0,
            collision_pairs: 3,
            contacts: 2,
            solved_constraints: 2,
        };
        a.accumulate(&b);
        assert!((a.broadphase_ms - 1.5).abs() < 1e-12);
        assert_eq!(a.collision_pairs, 13);
        assert_eq!(a.contacts, 8);
        assert_eq!(a.solved_constraints, 7);
    }
    #[test]
    fn stats_accumulate_saturates_instead_of_overflowing() {
        let mut a = PipelineStats {
            collision_pairs: u32::MAX - 1,
            contacts: u32::MAX,
            solved_constraints: 7,
            ..Default::default()
        };
        let b = PipelineStats {
            collision_pairs: 5,
            contacts: 1,
            solved_constraints: u32::MAX,
            ..Default::default()
        };
        a.accumulate(&b);
        assert_eq!(a.collision_pairs, u32::MAX);
        assert_eq!(a.contacts, u32::MAX);
        assert_eq!(a.solved_constraints, u32::MAX);
    }
    #[test]
    fn pipeline_cumulative_stats_grow() {
        let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
        let mut world = WorldState {
            positions: vec![0.0; 6],
            velocities: vec![0.0; 6],
            inverse_masses: vec![1.0, 1.0],
            ..Default::default()
        };
        let first = pipeline.step(&mut world, 0.016).expect("valid step");
        let second = pipeline.step(&mut world, 0.016).expect("valid step");
        assert!(pipeline.stats.total_time_ms >= 0.0);
        assert!(
            (pipeline.stats.total_time_ms - (first.total_time_ms + second.total_time_ms)).abs()
                < 1e-9
        );
        // Without a collision shape the two coincident bodies do not interact.
        assert_eq!(pipeline.stats.collision_pairs, 0);
        assert_eq!(pipeline.stats.solved_constraints, 0);
    }
    #[test]
    fn integration_stage_moves_body_downward() {
        let cfg = PipelineBuilder::new()
            .disable_stage(PipelineStage::BroadPhase)
            .disable_stage(PipelineStage::NarrowPhase)
            .disable_stage(PipelineStage::ConstraintSolve)
            .disable_stage(PipelineStage::PostProcess)
            .build()
            .config;
        let mut pipeline = PhysicsPipeline::new(cfg);
        let mut world = WorldState {
            positions: vec![0.0, 10.0, 0.0],
            velocities: vec![0.0, 0.0, 0.0],
            inverse_masses: vec![1.0],
            ..Default::default()
        };
        pipeline.step(&mut world, 1.0).expect("valid step");
        assert!(
            world.positions[1] < 10.0,
            "y should decrease due to gravity"
        );
        // Semi-implicit Euler: v = -g dt, y = y0 + v dt.
        assert!((world.velocities[1] + 9.81).abs() < 1e-12);
        assert!((world.positions[1] - (10.0 - 9.81)).abs() < 1e-12);
        assert_eq!(world.positions[0], 0.0);
        assert_eq!(world.positions[2], 0.0);
    }
    #[test]
    fn barrier_valid_order() {
        let b = ResourceBarrier::new(
            PipelineStage::BroadPhase,
            PipelineStage::NarrowPhase,
            "pair_buffer",
        );
        assert!(b.is_valid_order());
    }
    #[test]
    fn barrier_invalid_reverse_order() {
        let b = ResourceBarrier::new(PipelineStage::NarrowPhase, PipelineStage::BroadPhase, "bad");
        assert!(!b.is_valid_order());
    }
    #[test]
    fn barrier_set_validate_catches_bad() {
        let mut bs = BarrierSet::new();
        bs.add(ResourceBarrier::new(
            PipelineStage::BroadPhase,
            PipelineStage::NarrowPhase,
            "ok",
        ));
        bs.add(ResourceBarrier::new(
            PipelineStage::PostProcess,
            PipelineStage::Integration,
            "backwards",
        ));
        assert_eq!(bs.validate().len(), 1);
    }
    #[test]
    fn barrier_set_filter_by_stage() {
        let mut bs = BarrierSet::new();
        bs.add(ResourceBarrier::new(
            PipelineStage::BroadPhase,
            PipelineStage::NarrowPhase,
            "pair_buf",
        ));
        bs.add(ResourceBarrier::new(
            PipelineStage::NarrowPhase,
            PipelineStage::ConstraintSolve,
            "contact_buf",
        ));
        assert_eq!(bs.barriers_from(PipelineStage::BroadPhase).len(), 1);
        assert_eq!(bs.barriers_to(PipelineStage::ConstraintSolve).len(), 1);
        assert_eq!(bs.len(), 2);
    }
    #[test]
    fn async_queue_submit_and_flush() {
        let mut q = AsyncComputeQueue::new();
        assert!(q.is_idle());
        let p = ComputePipeline::new("test", "", "main");
        q.submit(DispatchBatch::new(p.clone(), 64));
        q.submit(DispatchBatch::new(p, 128));
        assert_eq!(q.pending(), 2);
        let executed = q.flush();
        assert_eq!(executed, 2);
        assert!(q.is_idle());
        assert_eq!(q.total_enqueued, 2);
        assert_eq!(q.total_executed, 2);
    }
    #[test]
    fn async_queue_flush_empty_returns_zero() {
        let mut q = AsyncComputeQueue::new();
        assert_eq!(q.flush(), 0);
    }
    #[test]
    fn async_queue_pending_decrements_on_flush() {
        let mut q = AsyncComputeQueue::new();
        let p = ComputePipeline::new("t", "", "main");
        for _ in 0..5 {
            q.submit(DispatchBatch::new(p.clone(), 32));
        }
        assert_eq!(q.pending(), 5);
        q.flush();
        assert_eq!(q.pending(), 0);
    }
    #[test]
    fn profiler_record_and_summary() {
        let mut prof = PipelineProfiler::new();
        prof.record("broadphase", 1.0);
        prof.record("broadphase", 3.0);
        let (mean, _std, n) = prof.summary("broadphase").unwrap();
        assert!((mean - 2.0).abs() < 1e-10);
        assert_eq!(n, 2);
    }
    #[test]
    fn profiler_summary_unknown_stage_is_none() {
        let prof = PipelineProfiler::new();
        assert!(prof.summary("nonexistent").is_none());
    }
    #[test]
    fn profiler_total_samples() {
        let mut prof = PipelineProfiler::new();
        prof.record("a", 1.0);
        prof.record("a", 2.0);
        prof.record("b", 3.0);
        assert_eq!(prof.total_samples(), 3);
    }
    #[test]
    fn profiler_stage_names_sorted() {
        let mut prof = PipelineProfiler::new();
        prof.record("zzz", 1.0);
        prof.record("aaa", 1.0);
        prof.record("mmm", 1.0);
        let names = prof.stage_names();
        assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
    }
    #[test]
    fn profiler_reset_clears_all() {
        let mut prof = PipelineProfiler::new();
        prof.record("broadphase", 1.5);
        prof.reset();
        assert_eq!(prof.total_samples(), 0);
        assert!(prof.stage_names().is_empty());
    }
    #[test]
    fn profiler_stddev_uniform_values() {
        let mut prof = PipelineProfiler::new();
        for _ in 0..4 {
            prof.record("stage", 5.0);
        }
        let (mean, std, _) = prof.summary("stage").unwrap();
        assert!((mean - 5.0).abs() < 1e-10);
        assert!(std < 1e-10, "stddev should be ~0 for identical samples");
    }
    #[test]
    fn pool_alloc_basic() {
        let mut pool = GpuMemoryPool::new(1024);
        let h = pool.alloc(256).expect("alloc should succeed");
        assert_eq!(h.1, 256);
        assert_eq!(pool.allocated, 256);
        assert_eq!(pool.free_space(), 768);
    }
    #[test]
    fn pool_alloc_and_free_round_trip() {
        let mut pool = GpuMemoryPool::new(1024);
        let h = pool.alloc(512).unwrap();
        pool.free(h.0, h.1).expect("free should succeed");
        assert!(pool.is_fully_free());
        assert_eq!(pool.fragmentation_count(), 1);
    }
    #[test]
    fn pool_alloc_exhaustion_returns_none() {
        let mut pool = GpuMemoryPool::new(100);
        let _ = pool.alloc(100).unwrap();
        assert!(
            pool.alloc(1).is_none(),
            "pool exhausted → alloc should fail"
        );
    }
    #[test]
    fn pool_multiple_allocs_and_frees() {
        let mut pool = GpuMemoryPool::new(1024);
        let h1 = pool.alloc(256).unwrap();
        let h2 = pool.alloc(256).unwrap();
        pool.free(h1.0, h1.1).unwrap();
        pool.free(h2.0, h2.1).unwrap();
        assert!(pool.is_fully_free());
    }
    #[test]
    fn pool_alloc_buffer_returns_correct_size() {
        let mut pool = GpuMemoryPool::new(2048);
        let (buf, handle) = pool
            .alloc_buffer("positions", 512, BufferUsage::Storage)
            .expect("alloc_buffer should succeed");
        assert_eq!(buf.len(), 512);
        assert_eq!(handle.1, 512);
    }
    #[test]
    fn resource_handle_from_alloc() {
        let handle = ResourceHandle::from_alloc((64, 128));
        assert_eq!(handle.offset, 64);
        assert_eq!(handle.size, 128);
    }
    #[test]
    fn pool_coalesces_free_blocks() {
        let mut pool = GpuMemoryPool::new(300);
        let h1 = pool.alloc(100).unwrap();
        let h2 = pool.alloc(100).unwrap();
        let h3 = pool.alloc(100).unwrap();
        pool.free(h1.0, h1.1).unwrap();
        pool.free(h2.0, h2.1).unwrap();
        pool.free(h3.0, h3.1).unwrap();
        assert_eq!(pool.fragmentation_count(), 1, "all blocks should coalesce");
    }
}
#[cfg(test)]
mod extended_pipeline_tests {

    use crate::pipeline::BarrierOptimizer;

    use crate::pipeline::ComputeOverlapScheduler;
    use crate::pipeline::ComputePipeline;

    use crate::pipeline::DispatchBatch;
    use crate::pipeline::FrameGraph;
    use crate::pipeline::FrameGraphPass;
    use crate::pipeline::GpuMemoryPool;
    use crate::pipeline::MultiQueueRecorder;

    use crate::pipeline::PipelineStage;
    use crate::pipeline::PipelineStatistics;
    use crate::pipeline::PipelineStats;
    use crate::pipeline::QueueType;
    use crate::pipeline::ResourceAliasingTracker;
    use crate::pipeline::ResourceBarrier;

    use crate::pipeline::StageTimer;
    use crate::pipeline::TimestampQuery;
    use crate::pipeline::TimestampQuerySet;
    use crate::pipeline::WorldState;
    #[test]
    fn timestamp_query_elapsed() {
        let q = TimestampQuery::new("broadphase", 1.0, 3.5);
        assert!((q.elapsed_ms() - 2.5).abs() < 1e-10);
    }
    #[test]
    fn timestamp_query_zero_duration() {
        let q = TimestampQuery::new("instant", 5.0, 5.0);
        assert!((q.elapsed_ms()).abs() < 1e-10);
    }
    #[test]
    fn query_set_slowest_pass() {
        let mut qs = TimestampQuerySet::new();
        qs.record(TimestampQuery::new("a", 0.0, 1.0));
        qs.record(TimestampQuery::new("b", 0.0, 5.0));
        qs.record(TimestampQuery::new("c", 0.0, 2.0));
        let slowest = qs.slowest_pass().unwrap();
        assert_eq!(slowest.label, "b");
    }
    #[test]
    fn query_set_total_elapsed() {
        let mut qs = TimestampQuerySet::new();
        qs.record(TimestampQuery::new("x", 0.0, 1.0));
        qs.record(TimestampQuery::new("y", 0.0, 2.0));
        assert!((qs.total_elapsed_ms() - 3.0).abs() < 1e-10);
    }
    #[test]
    fn query_set_empty_slowest_none() {
        let qs = TimestampQuerySet::new();
        assert!(qs.slowest_pass().is_none());
    }
    #[test]
    fn query_set_clear() {
        let mut qs = TimestampQuerySet::new();
        qs.record(TimestampQuery::new("a", 0.0, 1.0));
        qs.clear();
        assert!(qs.queries().is_empty());
    }
    #[test]
    fn pipeline_stats_arithmetic_intensity() {
        let s = PipelineStatistics {
            cs_invocations: 1024,
            workgroups_dispatched: 16,
            flops: 2048,
            bytes_read: 512,
            bytes_written: 512,
        };
        let ai = s.arithmetic_intensity();
        assert!((ai - 2.0).abs() < 1e-10, "ai = {ai}");
    }
    #[test]
    fn pipeline_stats_zero_bytes() {
        let s = PipelineStatistics::default();
        assert!((s.arithmetic_intensity()).abs() < 1e-10);
    }
    #[test]
    fn pipeline_stats_bandwidth_utilisation() {
        let s = PipelineStatistics {
            bytes_read: 500_000_000,
            bytes_written: 500_000_000,
            ..Default::default()
        };
        let util = s.bandwidth_utilization(1_000_000_000.0, 1.0);
        assert!((util - 1.0).abs() < 1e-6, "util = {util}");
    }
    #[test]
    fn multi_queue_recorder_submit_and_flush() {
        let mut rec = MultiQueueRecorder::new();
        let p = ComputePipeline::new("t", "", "main");
        rec.submit(DispatchBatch::new(p.clone(), 64), QueueType::Main);
        rec.submit(DispatchBatch::new(p.clone(), 64), QueueType::AsyncCompute);
        rec.submit(DispatchBatch::new(p, 64), QueueType::Transfer);
        assert_eq!(rec.pending_total(), 3);
        let flushed = rec.flush_all();
        assert_eq!(flushed, 3);
        assert_eq!(rec.pending_total(), 0);
    }
    #[test]
    fn multi_queue_recorder_total_recorded() {
        let mut rec = MultiQueueRecorder::new();
        let p = ComputePipeline::new("t", "", "main");
        for _ in 0..5 {
            rec.submit(DispatchBatch::new(p.clone(), 32), QueueType::Main);
        }
        assert_eq!(rec.total_recorded, 5);
    }
    #[test]
    fn barrier_optimizer_removes_duplicates() {
        let barriers = vec![
            ResourceBarrier::new(PipelineStage::BroadPhase, PipelineStage::NarrowPhase, "buf"),
            ResourceBarrier::new(PipelineStage::BroadPhase, PipelineStage::NarrowPhase, "buf"),
        ];
        let opt = BarrierOptimizer::optimize(&barriers);
        assert_eq!(opt.len(), 1, "duplicate should be removed");
        assert_eq!(BarrierOptimizer::savings(&barriers), 1);
    }
    #[test]
    fn barrier_optimizer_preserves_different_resources() {
        let barriers = vec![
            ResourceBarrier::new(
                PipelineStage::BroadPhase,
                PipelineStage::NarrowPhase,
                "buf_a",
            ),
            ResourceBarrier::new(
                PipelineStage::BroadPhase,
                PipelineStage::NarrowPhase,
                "buf_b",
            ),
        ];
        let opt = BarrierOptimizer::optimize(&barriers);
        assert_eq!(opt.len(), 2);
    }
    #[test]
    fn barrier_optimizer_empty_input() {
        let opt = BarrierOptimizer::optimize(&[]);
        assert!(opt.is_empty());
        assert_eq!(BarrierOptimizer::savings(&[]), 0);
    }
    #[test]
    fn aliasing_tracker_detects_shared_allocation() {
        let mut t = ResourceAliasingTracker::new();
        t.track("shadow_map", 0, 1024);
        t.track("gbuffer_depth", 0, 1024);
        assert!(t.are_aliased("shadow_map", "gbuffer_depth"));
    }
    #[test]
    fn aliasing_tracker_no_alias() {
        let mut t = ResourceAliasingTracker::new();
        t.track("a", 0, 512);
        t.track("b", 512, 512);
        assert!(!t.are_aliased("a", "b"));
    }
    #[test]
    fn aliasing_tracker_aliases_for() {
        let mut t = ResourceAliasingTracker::new();
        t.track("x", 100, 200);
        t.track("y", 100, 200);
        let aliases = t.aliases_for(100, 200);
        assert_eq!(aliases.len(), 2);
    }
    #[test]
    fn aliasing_tracker_counts() {
        let mut t = ResourceAliasingTracker::new();
        t.track("a", 0, 256);
        t.track("b", 256, 256);
        t.track("c", 0, 256);
        assert_eq!(t.allocation_count(), 2);
        assert_eq!(t.total_resource_registrations(), 3);
    }
    #[test]
    fn overlap_scheduler_critical_and_background() {
        let mut sched = ComputeOverlapScheduler::new();
        let p = ComputePipeline::new("t", "", "main");
        sched.submit_critical(DispatchBatch::new(p.clone(), 64));
        sched.submit_background(DispatchBatch::new(p, 64));
        assert!(sched.has_pending());
        let n = sched.end_frame();
        assert_eq!(n, 2);
        assert!(!sched.has_pending());
        assert_eq!(sched.critical_count, 0);
    }
    #[test]
    fn overlap_scheduler_empty_frame() {
        let mut sched = ComputeOverlapScheduler::new();
        assert!(!sched.has_pending());
        let n = sched.end_frame();
        assert_eq!(n, 0);
    }
    #[test]
    fn frame_graph_writers_and_readers() {
        let mut fg = FrameGraph::new();
        fg.add_pass(FrameGraphPass::new("shadow_pass", QueueType::Main).writes("shadow_map"));
        fg.add_pass(FrameGraphPass::new("lighting_pass", QueueType::Main).reads("shadow_map"));
        let writers = fg.writers_of("shadow_map");
        assert_eq!(writers.len(), 1);
        assert_eq!(writers[0].name, "shadow_pass");
        let readers = fg.readers_of("shadow_map");
        assert_eq!(readers.len(), 1);
        assert_eq!(readers[0].name, "lighting_pass");
    }
    #[test]
    fn frame_graph_validate_valid_deps() {
        let mut fg = FrameGraph::new();
        fg.add_pass(FrameGraphPass::new("pass_a", QueueType::Main));
        fg.add_pass(FrameGraphPass::new("pass_b", QueueType::Main).depends_on("pass_a"));
        assert!(
            fg.validate_dependencies().is_empty(),
            "valid deps should produce no errors"
        );
    }
    #[test]
    fn frame_graph_validate_invalid_dep() {
        let mut fg = FrameGraph::new();
        fg.add_pass(FrameGraphPass::new("pass_b", QueueType::Main).depends_on("nonexistent"));
        let errors = fg.validate_dependencies();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("nonexistent"));
    }
    #[test]
    fn frame_graph_async_pass_count() {
        let mut fg = FrameGraph::new();
        fg.add_pass(FrameGraphPass::new("main1", QueueType::Main));
        fg.add_pass(FrameGraphPass::new("async1", QueueType::AsyncCompute));
        fg.add_pass(FrameGraphPass::new("async2", QueueType::AsyncCompute));
        assert_eq!(fg.async_pass_count(), 2);
    }
    #[test]
    fn frame_graph_pass_names_order() {
        let mut fg = FrameGraph::new();
        fg.add_pass(FrameGraphPass::new("z", QueueType::Main));
        fg.add_pass(FrameGraphPass::new("a", QueueType::Main));
        let names = fg.pass_names();
        assert_eq!(names, vec!["z", "a"]);
    }
    #[test]
    fn pool_free_out_of_bounds_returns_err() {
        let mut pool = GpuMemoryPool::new(100);
        assert!(
            pool.free(200, 10).is_err(),
            "out-of-bounds free should fail"
        );
    }
    #[test]
    fn pool_fragmentation_after_partial_free() {
        let mut pool = GpuMemoryPool::new(300);
        let h1 = pool.alloc(100).unwrap();
        let _h2 = pool.alloc(100).unwrap();
        let _h3 = pool.alloc(100).unwrap();
        pool.free(h1.0, h1.1).unwrap();
        assert!(pool.fragmentation_count() >= 1);
    }
    #[test]
    fn stage_timer_records_elapsed() {
        let mut timer = StageTimer::start();
        timer.stop();
        assert!(timer.elapsed_ms >= 0.0, "elapsed should be non-negative");
    }
    #[test]
    fn pipeline_stats_stage_total() {
        let s = PipelineStats {
            broadphase_ms: 1.0,
            narrowphase_ms: 2.0,
            constraint_ms: 3.0,
            integration_ms: 4.0,
            postprocess_ms: 5.0,
            ..Default::default()
        };
        assert!((s.stage_total_ms() - 15.0).abs() < 1e-10);
    }
    #[test]
    fn world_state_body_count() {
        let w = WorldState {
            positions: vec![0.0; 9],
            velocities: vec![0.0; 9],
            inverse_masses: vec![1.0, 1.0, 1.0],
            ..Default::default()
        };
        assert_eq!(w.body_count(), 3);
    }
}
