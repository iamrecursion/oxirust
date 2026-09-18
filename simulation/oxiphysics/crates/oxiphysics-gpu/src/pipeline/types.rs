//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::functions::*;
use std::fmt;
use std::time::Instant;

/// Schedules compute dispatches to maximise GPU utilisation through overlapping.
///
/// Classifies dispatches as either "latency-critical" (must finish before the
/// next frame) or "background" (can run async on a secondary queue).
#[derive(Debug)]
pub struct ComputeOverlapScheduler {
    /// Number of latency-critical dispatches submitted this frame.
    pub critical_count: usize,
    /// Number of background dispatches submitted this frame.
    pub background_count: usize,
    pub(super) recorder: MultiQueueRecorder,
}
impl ComputeOverlapScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            critical_count: 0,
            background_count: 0,
            recorder: MultiQueueRecorder::new(),
        }
    }
    /// Submit a latency-critical dispatch (→ main queue).
    pub fn submit_critical(&mut self, batch: DispatchBatch) {
        self.critical_count += 1;
        self.recorder.submit(batch, QueueType::Main);
    }
    /// Submit a background dispatch (→ async compute queue).
    pub fn submit_background(&mut self, batch: DispatchBatch) {
        self.background_count += 1;
        self.recorder.submit(batch, QueueType::AsyncCompute);
    }
    /// Execute all pending work and reset frame counters.
    pub fn end_frame(&mut self) -> usize {
        let n = self.recorder.flush_all();
        self.critical_count = 0;
        self.background_count = 0;
        n
    }
    /// Returns `true` if there is any pending work.
    pub fn has_pending(&self) -> bool {
        self.recorder.pending_total() > 0
    }
}
/// Default number of Gauss-Seidel iterations of the contact solver per sub-step.
pub const DEFAULT_SOLVER_ITERATIONS: u32 = 8;
/// Default Baumgarte factor of the contact solver: the fraction of the
/// penetration (beyond the slop) that is removed per time step.
pub const DEFAULT_BAUMGARTE: f64 = 0.2;
/// Configuration for the high-level physics pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Which stages are active.  Stages not in this list are skipped.
    pub enabled_stages: Vec<PipelineStage>,
    /// Number of sub-steps per call to [`PhysicsPipeline::step`].
    pub substeps: u32,
    /// Whether the broad phase sorts its sweep axis on the GPU.
    ///
    /// When `true` the bodies are ordered by the multi-pass radix sort of
    /// [`crate::gpu_radix::radix_sort_pairs_gpu`]; that sort falls back to the CPU
    /// by itself when no GPU adapter is available (or the `wgpu-backend` feature is
    /// disabled).  The sorted order, and therefore the pairs, are identical to those
    /// of the CPU sort.  Narrow phase, solver, integration and post-processing
    /// always run on the CPU.
    pub use_gpu: bool,
    /// Gauss-Seidel iterations of the contact solver per sub-step (`0` runs one).
    pub solver_iterations: u32,
    /// Coefficient of restitution of every contact, clamped to `[0, 1]`
    /// (`NaN` counts as `0`).  Contacts that close slower than two steps of
    /// gravity are resting contacts and never bounce.
    pub restitution: f64,
    /// Baumgarte factor of the contact solver, clamped to `[0, 1]` (`NaN` counts
    /// as `0`): the contact solver pushes penetrating bodies apart at
    /// `baumgarte / dt` times the penetration depth beyond a small slop.
    pub baumgarte: f64,
}
impl PipelineConfig {
    /// Default config: all stages enabled, 1 substep, CPU backend, perfectly
    /// inelastic contacts ([`DEFAULT_SOLVER_ITERATIONS`] solver iterations,
    /// [`DEFAULT_BAUMGARTE`] stabilisation).
    pub fn new() -> Self {
        Self {
            enabled_stages: PipelineStage::all_in_order().to_vec(),
            substeps: 1,
            use_gpu: false,
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            restitution: 0.0,
            baumgarte: DEFAULT_BAUMGARTE,
        }
    }
    /// Returns `true` if `stage` is in the enabled list.
    pub fn is_enabled(&self, stage: PipelineStage) -> bool {
        self.enabled_stages.contains(&stage)
    }
}
/// Orchestrates a complete simulation step through all enabled pipeline stages.
///
/// The stage execution order is guaranteed:
/// `BroadPhase → NarrowPhase → ConstraintSolve → Integration → PostProcess`
///
/// Every body of the [`WorldState`] with a positive radius is a sphere:
///
/// 1. **BroadPhase** finds the pairs of spheres whose bounding boxes overlap with a
///    sort-and-sweep along the axis the bodies spread over the most.  Pairs of two
///    immovable bodies are never reported.
/// 2. **NarrowPhase** keeps the pairs whose spheres really penetrate and builds a
///    [`SphereContact`] (normal, depth, contact point) for each of them.
/// 3. **ConstraintSolve** removes the approaching velocity of every contact with
///    sequential impulses (accumulated, non-negative normal impulses, restitution
///    and Baumgarte stabilisation, see [`PipelineConfig`]).
/// 4. **Integration** applies gravity and advances the positions of the dynamic
///    bodies with semi-implicit Euler.
/// 5. **PostProcess** rejects non-finite state: the step fails with a
///    [`PipelineError`] and the world is restored to the state it had before the
///    step.
#[derive(Debug)]
pub struct PhysicsPipeline {
    /// Pipeline configuration.
    pub config: PipelineConfig,
    /// Cumulative stats across all steps since creation.
    pub stats: PipelineStats,
    /// Broad-phase pairs `(a, b)` with `a < b` of the most recent sub-step.
    pairs: Vec<(usize, usize)>,
    /// Narrow-phase contacts of the most recent sub-step.
    contacts: Vec<SphereContact>,
    /// Reusable broad-phase sort buffers.
    sweep: BroadphaseScratch,
    /// Positions at the start of the current step, restored when it is rejected.
    rollback_positions: Vec<f64>,
    /// Velocities at the start of the current step, restored when it is rejected.
    rollback_velocities: Vec<f64>,
}
impl PhysicsPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            stats: PipelineStats::default(),
            pairs: Vec::new(),
            contacts: Vec::new(),
            sweep: BroadphaseScratch::default(),
            rollback_positions: Vec::new(),
            rollback_velocities: Vec::new(),
        }
    }
    /// Broad-phase pairs `(a, b)` (with `a < b`, in ascending order) of the most
    /// recent sub-step: bodies whose bounding boxes overlap and that are not both
    /// immovable.  Empty before the first step and after a rejected step.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }
    /// Narrow-phase contacts of the most recent sub-step, carrying the normal
    /// impulses the solver applied.  Empty before the first step and after a
    /// rejected step.
    pub fn contacts(&self) -> &[SphereContact] {
        &self.contacts
    }
    /// Run one full simulation step (all enabled stages, for `substeps`
    /// iterations) over `world_state` with time step `dt`.
    ///
    /// Returns the [`PipelineStats`] for this step.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::InvalidTimeStep`] if `dt` is negative, infinite or `NaN`;
    /// * [`PipelineError::LayoutMismatch`] if the arrays of `world_state` disagree
    ///   about the number of bodies (see [`WorldState::validate_layout`]);
    /// * [`PipelineError::NonFiniteState`] or [`PipelineError::NegativeInverseMass`]
    ///   if the [`PipelineStage::PostProcess`] stage finds invalid state.
    ///
    /// The step is atomic: when it fails, `world_state` is exactly as it was before
    /// the call and the cumulative [`PhysicsPipeline::stats`] are unchanged.  Without
    /// the post-process stage no state is validated and no rollback is needed.
    pub fn step(
        &mut self,
        world_state: &mut WorldState,
        dt: f64,
    ) -> Result<PipelineStats, PipelineError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(PipelineError::InvalidTimeStep { dt });
        }
        world_state.validate_layout()?;
        let step_start = Instant::now();
        let validates_state = self.config.is_enabled(PipelineStage::PostProcess);
        if validates_state {
            self.rollback_positions.clone_from(&world_state.positions);
            self.rollback_velocities.clone_from(&world_state.velocities);
        }
        let mut step_stats = PipelineStats::default();
        let sub_dt = if self.config.substeps > 0 {
            dt / self.config.substeps as f64
        } else {
            dt
        };
        for _ in 0..self.config.substeps.max(1) {
            match self.run_stages(world_state, sub_dt) {
                Ok(sub_stats) => step_stats.accumulate(&sub_stats),
                Err(error) => {
                    if validates_state {
                        world_state.positions.clone_from(&self.rollback_positions);
                        world_state.velocities.clone_from(&self.rollback_velocities);
                    }
                    self.pairs.clear();
                    self.contacts.clear();
                    return Err(error);
                }
            }
        }
        step_stats.total_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        self.stats.accumulate(&step_stats);
        Ok(step_stats)
    }
    /// Run all enabled stages once for a single sub-step.
    ///
    /// Each stage consumes what the stage before it produced in this sub-step; a
    /// disabled stage produces nothing (without the broad phase there are no pairs,
    /// without the narrow phase no contacts).
    fn run_stages(
        &mut self,
        world_state: &mut WorldState,
        dt: f64,
    ) -> Result<PipelineStats, PipelineError> {
        let mut stats = PipelineStats::default();
        self.pairs.clear();
        self.contacts.clear();
        for stage in PipelineStage::all_in_order() {
            if !self.config.is_enabled(stage) {
                continue;
            }
            let mut timer = StageTimer::start();
            match stage {
                PipelineStage::BroadPhase => {
                    run_broadphase(
                        world_state,
                        self.config.use_gpu,
                        &mut self.sweep,
                        &mut self.pairs,
                    );
                    stats.collision_pairs = saturating_count(self.pairs.len());
                }
                PipelineStage::NarrowPhase => {
                    run_narrowphase(world_state, &self.pairs, &mut self.contacts);
                    stats.contacts = saturating_count(self.contacts.len());
                }
                PipelineStage::ConstraintSolve => {
                    let settings = SolverSettings::from_config(&self.config);
                    let solved =
                        run_constraint_solve(world_state, &mut self.contacts, settings, dt);
                    stats.solved_constraints = saturating_count(solved);
                }
                PipelineStage::Integration => {
                    run_integration(world_state, dt);
                }
                PipelineStage::PostProcess => {
                    run_postprocess(world_state)?;
                }
            }
            timer.stop();
            match stage {
                PipelineStage::BroadPhase => stats.broadphase_ms += timer.elapsed_ms,
                PipelineStage::NarrowPhase => stats.narrowphase_ms += timer.elapsed_ms,
                PipelineStage::ConstraintSolve => stats.constraint_ms += timer.elapsed_ms,
                PipelineStage::Integration => stats.integration_ms += timer.elapsed_ms,
                PipelineStage::PostProcess => stats.postprocess_ms += timer.elapsed_ms,
            }
        }
        Ok(stats)
    }
}
/// GPU compute pipeline descriptor (wgpu-agnostic, usable in unit tests).
#[derive(Debug, Clone)]
pub struct ComputePipeline {
    /// Human-readable label for debugging.
    pub label: String,
    /// Raw WGSL shader source.
    pub shader_source: String,
    /// Name of the entry-point function inside the shader.
    pub entry_point: String,
    /// Workgroup size declared in the shader `[X, Y, Z]`.
    pub workgroup_size: [u32; 3],
}
impl ComputePipeline {
    /// Create a new pipeline descriptor.
    ///
    /// `workgroup_size` defaults to `[64, 1, 1]` (matching the shaders in
    /// [`crate::shaders`]).
    pub fn new(label: &str, shader: &str, entry_point: &str) -> Self {
        Self {
            label: label.to_owned(),
            shader_source: shader.to_owned(),
            entry_point: entry_point.to_owned(),
            workgroup_size: [64, 1, 1],
        }
    }
    /// Compute the dispatch grid dimensions needed to cover `n_items` work
    /// items.
    ///
    /// Returns `[ceil(n_items / workgroup_size[0\]), 1, 1]`.
    pub fn workgroups_needed(&self, n_items: u32) -> [u32; 3] {
        let x = n_items.div_ceil(self.workgroup_size[0]);
        [x, 1, 1]
    }
}
/// Describes a memory/execution barrier between two pipeline stages.
///
/// On a real GPU this would translate to a `vkCmdPipelineBarrier` or
/// `wgpu::CommandEncoder::insert_debug_marker`.  Here it is a pure data
/// type used to validate that the user has not forgotten to insert barriers
/// between dependent passes.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceBarrier {
    /// The stage that *writes* the resource.
    pub src_stage: PipelineStage,
    /// The stage that *reads* the resource.
    pub dst_stage: PipelineStage,
    /// Human-readable name of the protected resource.
    pub resource_name: String,
}
impl ResourceBarrier {
    /// Create a new barrier descriptor.
    pub fn new(src: PipelineStage, dst: PipelineStage, name: &str) -> Self {
        Self {
            src_stage: src,
            dst_stage: dst,
            resource_name: name.to_owned(),
        }
    }
    /// Returns `true` if the barrier is between stages in the correct
    /// dependency order (src must come before dst).
    pub fn is_valid_order(&self) -> bool {
        self.src_stage < self.dst_stage
    }
}
/// An opaque handle to a pool-allocated GPU resource.
///
/// Stores the `(offset, size)` pair returned by [`GpuMemoryPool::alloc`] so
/// that the caller can later free the resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceHandle {
    /// Offset into the pool's backing store.
    pub offset: usize,
    /// Size of the allocation in `f32` elements.
    pub size: usize,
}
impl ResourceHandle {
    /// Create a handle from a raw `(offset, size)` pair.
    pub fn from_alloc(alloc: (usize, usize)) -> Self {
        Self {
            offset: alloc.0,
            size: alloc.1,
        }
    }
}
/// A simple frame-graph pass descriptor.
///
/// Frame graphs describe the rendering / compute workload as a DAG of passes,
/// enabling automatic barrier insertion and resource aliasing.
#[derive(Debug, Clone)]
pub struct FrameGraphPass {
    /// Unique name for this pass.
    pub name: String,
    /// Names of resources this pass reads.
    pub reads: Vec<String>,
    /// Names of resources this pass writes.
    pub writes: Vec<String>,
    /// Names of passes that must complete before this one.
    pub dependencies: Vec<String>,
    /// Which queue this pass runs on.
    pub queue: QueueType,
}
impl FrameGraphPass {
    /// Create a new pass with no dependencies.
    pub fn new(name: impl Into<String>, queue: QueueType) -> Self {
        Self {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            dependencies: Vec::new(),
            queue,
        }
    }
    /// Add a resource read.
    pub fn reads(mut self, resource: impl Into<String>) -> Self {
        self.reads.push(resource.into());
        self
    }
    /// Add a resource write.
    pub fn writes(mut self, resource: impl Into<String>) -> Self {
        self.writes.push(resource.into());
        self
    }
    /// Add a pass dependency.
    pub fn depends_on(mut self, pass: impl Into<String>) -> Self {
        self.dependencies.push(pass.into());
        self
    }
}
/// A minimal frame graph that can validate pass dependencies.
#[derive(Debug, Default)]
pub struct FrameGraph {
    pub(super) passes: Vec<FrameGraphPass>,
}
impl FrameGraph {
    /// Create an empty frame graph.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a pass.
    pub fn add_pass(&mut self, pass: FrameGraphPass) {
        self.passes.push(pass);
    }
    /// Return all pass names in insertion order.
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name.as_str()).collect()
    }
    /// Return all passes that write to `resource`.
    pub fn writers_of(&self, resource: &str) -> Vec<&FrameGraphPass> {
        self.passes
            .iter()
            .filter(|p| p.writes.iter().any(|w| w == resource))
            .collect()
    }
    /// Return all passes that read from `resource`.
    pub fn readers_of(&self, resource: &str) -> Vec<&FrameGraphPass> {
        self.passes
            .iter()
            .filter(|p| p.reads.iter().any(|r| r == resource))
            .collect()
    }
    /// Validate that all declared dependencies reference existing passes.
    /// Returns a list of invalid dependency references.
    pub fn validate_dependencies(&self) -> Vec<String> {
        let names: std::collections::HashSet<&str> =
            self.passes.iter().map(|p| p.name.as_str()).collect();
        let mut errors = Vec::new();
        for pass in &self.passes {
            for dep in &pass.dependencies {
                if !names.contains(dep.as_str()) {
                    errors.push(format!("{}: unknown dependency '{}'", pass.name, dep));
                }
            }
        }
        errors
    }
    /// Count how many passes run on the async compute queue.
    pub fn async_pass_count(&self) -> usize {
        self.passes
            .iter()
            .filter(|p| p.queue == QueueType::AsyncCompute)
            .count()
    }
}
/// A distinct stage in the physics simulation pipeline.
///
/// Stages run in dependency order:
/// `BroadPhase → NarrowPhase → ConstraintSolve → Integration → PostProcess`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PipelineStage {
    /// Broad-phase collision detection (sort-and-sweep over the bounding boxes of
    /// the body spheres).
    BroadPhase,
    /// Narrow-phase contact generation (sphere-sphere normal and depth).
    NarrowPhase,
    /// Constraint / impulse solver (sequential-impulse non-penetration contacts).
    ConstraintSolve,
    /// Velocity and position integration (gravity, semi-implicit Euler).
    Integration,
    /// Post-processing: rejects non-finite state and rolls the step back.
    PostProcess,
}
impl PipelineStage {
    /// Ordered list of all stages, from first to last.
    pub fn all_in_order() -> [PipelineStage; 5] {
        [
            PipelineStage::BroadPhase,
            PipelineStage::NarrowPhase,
            PipelineStage::ConstraintSolve,
            PipelineStage::Integration,
            PipelineStage::PostProcess,
        ]
    }
}
/// A single compute dispatch pass: one pipeline plus its bound buffers.
#[derive(Debug)]
pub struct DispatchBatch {
    /// The pipeline to dispatch.
    pub pipeline: ComputePipeline,
    /// Buffers bound to this pass, in binding-slot order.
    pub bindings: Vec<CpuBuffer>,
    /// The dispatch grid dimensions `[X, Y, Z]`.
    pub dispatch_dims: [u32; 3],
}
impl DispatchBatch {
    /// Create a new dispatch batch, computing the required workgroup grid from
    /// `workitems`.
    pub fn new(pipeline: ComputePipeline, workitems: u32) -> Self {
        let dispatch_dims = pipeline.workgroups_needed(workitems);
        Self {
            pipeline,
            bindings: Vec::new(),
            dispatch_dims,
        }
    }
    /// Append `buffer` to the binding list (next available slot).
    pub fn bind(&mut self, buffer: CpuBuffer) {
        self.bindings.push(buffer);
    }
}
/// Models an async compute queue: a FIFO of [`DispatchBatch`]es that can be
/// submitted independently of the graphics queue.
///
/// On a GPU this enables overlapping async compute with graphics work.
/// Here it is a simple in-memory queue used to test scheduling logic.
pub struct AsyncComputeQueue {
    pub(super) queue: std::collections::VecDeque<DispatchBatch>,
    /// Total number of batches ever enqueued (for profiling).
    pub total_enqueued: usize,
    /// Total number of batches ever drained (executed).
    pub total_executed: usize,
}
impl AsyncComputeQueue {
    /// Create an empty async compute queue.
    pub fn new() -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
            total_enqueued: 0,
            total_executed: 0,
        }
    }
    /// Enqueue a dispatch batch.
    pub fn submit(&mut self, batch: DispatchBatch) {
        self.total_enqueued += 1;
        self.queue.push_back(batch);
    }
    /// Execute (drain) all pending batches and return the count executed.
    ///
    /// In a real backend this would flush the command buffer to the GPU.
    /// Here it simply clears the queue and updates counters.
    pub fn flush(&mut self) -> usize {
        let n = self.queue.len();
        self.total_executed += n;
        self.queue.clear();
        n
    }
    /// Number of batches currently waiting in the queue.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }
    /// Returns `true` when there are no pending batches.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty()
    }
}
/// A dispatched batch annotated with its target queue.
#[derive(Debug)]
pub struct MultiQueueBatch {
    /// The dispatch batch.
    pub batch: DispatchBatch,
    /// Which queue to submit to.
    pub queue: QueueType,
    /// Semaphore/fence dependency: must wait for this frame's signal.
    pub wait_frame: u64,
}
/// Minimal world state passed to [`PhysicsPipeline::step`].
///
/// Uses pure f64 arrays (no nalgebra) compatible with GPU buffer uploads.  Every
/// body is a sphere; the number of bodies is the length of `inverse_masses`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorldState {
    /// Flat position array `[x0, y0, z0, x1, y1, z1, ...]` of the sphere centres.
    pub positions: Vec<f64>,
    /// Flat velocity array `[vx0, vy0, vz0, ...]`.
    pub velocities: Vec<f64>,
    /// Inverse masses `[inv_m0, inv_m1, ...]`.  A body with an inverse mass of
    /// `0.0` is immovable: it is never integrated and never pushed by contacts.
    pub inverse_masses: Vec<f64>,
    /// Collision sphere radius of every body `[r0, r1, ...]`.
    ///
    /// The array is either empty (no body has a collision shape) or holds one entry
    /// per body.  A body whose radius is not positive has no collision shape and
    /// never takes part in a contact.
    pub radii: Vec<f64>,
}
impl WorldState {
    /// Number of bodies in the world state.
    pub fn body_count(&self) -> usize {
        self.inverse_masses.len()
    }
    /// Append a spherical body and return its index.
    ///
    /// `radius <= 0` creates a body without a collision shape.  The arrays must
    /// already be consistent (see [`WorldState::validate_layout`]); an empty
    /// `radii` array is extended with shapeless entries for the existing bodies.
    pub fn add_body(
        &mut self,
        position: [f64; 3],
        velocity: [f64; 3],
        inverse_mass: f64,
        radius: f64,
    ) -> usize {
        let body = self.body_count();
        if self.radii.len() < body {
            self.radii.resize(body, 0.0);
        }
        self.positions.extend_from_slice(&position);
        self.velocities.extend_from_slice(&velocity);
        self.inverse_masses.push(inverse_mass);
        self.radii.push(radius);
        body
    }
    /// Check that the per-body arrays agree about the number of bodies:
    /// `positions` and `velocities` hold three values per body and `radii` is
    /// either empty or holds one value per body.
    ///
    /// # Errors
    ///
    /// [`PipelineError::LayoutMismatch`] naming the first offending array.
    pub fn validate_layout(&self) -> Result<(), PipelineError> {
        let bodies = self.body_count();
        let vec3_len = bodies.saturating_mul(3);
        let checks = [
            ("positions", vec3_len, self.positions.len()),
            ("velocities", vec3_len, self.velocities.len()),
        ];
        for (array, expected, actual) in checks {
            if actual != expected {
                return Err(PipelineError::LayoutMismatch {
                    array,
                    expected,
                    actual,
                    bodies,
                });
            }
        }
        if !self.radii.is_empty() && self.radii.len() != bodies {
            return Err(PipelineError::LayoutMismatch {
                array: "radii",
                expected: bodies,
                actual: self.radii.len(),
                bodies,
            });
        }
        Ok(())
    }
    /// Centre of `body`, or `None` if the body does not exist or a coordinate is
    /// not finite.
    pub fn position(&self, body: usize) -> Option<[f64; 3]> {
        finite_triple(&self.positions, body)
    }
    /// Velocity of `body`, or `None` if the body does not exist or a component is
    /// not finite.
    pub fn velocity(&self, body: usize) -> Option<[f64; 3]> {
        finite_triple(&self.velocities, body)
    }
    /// Collision radius of `body`, or `None` if the body does not exist or has no
    /// collision shape (no radius entry, or a radius that is not finite and
    /// positive).
    pub fn collision_radius(&self, body: usize) -> Option<f64> {
        self.radii
            .get(body)
            .copied()
            .filter(|radius| radius.is_finite() && *radius > 0.0)
    }
    /// Inverse mass of `body`; `0.0` for an immovable body and for a body that
    /// does not exist or whose inverse mass is not a finite positive number.
    pub fn inverse_mass(&self, body: usize) -> f64 {
        self.inverse_masses
            .get(body)
            .copied()
            .filter(|inverse_mass| inverse_mass.is_finite() && *inverse_mass > 0.0)
            .unwrap_or(0.0)
    }
    /// Returns `true` if `body` can move, i.e. its inverse mass is finite and
    /// positive.
    pub fn is_dynamic(&self, body: usize) -> bool {
        self.inverse_mass(body) > 0.0
    }
    /// Add `delta` to the velocity of `body`; a body without a velocity entry is
    /// left alone.
    pub(super) fn add_to_velocity(&mut self, body: usize, delta: [f64; 3]) {
        if let Some(velocity) = triple_range(body).and_then(|range| self.velocities.get_mut(range))
        {
            for (component, change) in velocity.iter_mut().zip(delta) {
                *component += change;
            }
        }
    }
}
/// Index range of the three values of `body` in a flat per-body vector array.
fn triple_range(body: usize) -> Option<std::ops::Range<usize>> {
    let start = body.checked_mul(3)?;
    let end = start.checked_add(3)?;
    Some(start..end)
}
/// The three values of `body` in a flat per-body vector array, if all are finite.
fn finite_triple(values: &[f64], body: usize) -> Option<[f64; 3]> {
    match values.get(triple_range(body)?)? {
        &[x, y, z] if x.is_finite() && y.is_finite() && z.is_finite() => Some([x, y, z]),
        _ => None,
    }
}
/// Convert a count to the `u32` of [`PipelineStats`], saturating at `u32::MAX`.
fn saturating_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}
/// A penetrating sphere-sphere contact generated by the narrow phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereContact {
    /// Index of the first body (smaller than `body_b`).
    pub body_a: usize,
    /// Index of the second body.
    pub body_b: usize,
    /// Unit contact normal, pointing from body `a` towards body `b`.  When the two
    /// centres coincide the direction is undefined and `[1, 0, 0]` is used.
    pub normal: [f64; 3],
    /// Contact point: the middle of the overlap region on the line of centres.
    pub point: [f64; 3],
    /// Penetration depth `r_a + r_b - |c_b - c_a|`; always positive.
    pub depth: f64,
    /// Normal impulse the constraint solver applied to the pair (momentum
    /// transferred along `normal`; never negative).  `0.0` until the
    /// [`PipelineStage::ConstraintSolve`] stage has run.
    pub normal_impulse: f64,
}
/// Per-body quantity of a [`WorldState`] named by a [`PipelineError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateQuantity {
    /// Centre position.
    Position,
    /// Linear velocity.
    Velocity,
    /// Collision sphere radius.
    Radius,
    /// Inverse mass.
    InverseMass,
}
impl fmt::Display for StateQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            StateQuantity::Position => "position",
            StateQuantity::Velocity => "velocity",
            StateQuantity::Radius => "radius",
            StateQuantity::InverseMass => "inverse mass",
        };
        f.write_str(name)
    }
}
/// Why [`PhysicsPipeline::step`] rejected a step.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError {
    /// The time step is negative, infinite or `NaN`.
    #[error("time step {dt} is not a finite non-negative number")]
    InvalidTimeStep {
        /// The offending time step.
        dt: f64,
    },
    /// The per-body arrays of the [`WorldState`] disagree about the body count.
    #[error(
        "world state layout mismatch: `{array}` holds {actual} values but {expected} are required for {bodies} bodies"
    )]
    LayoutMismatch {
        /// Name of the offending array.
        array: &'static str,
        /// Required length.
        expected: usize,
        /// Actual length.
        actual: usize,
        /// Number of bodies (length of `inverse_masses`).
        bodies: usize,
    },
    /// A per-body quantity is `NaN` or infinite.
    #[error("body {body} has a non-finite {quantity}")]
    NonFiniteState {
        /// Index of the offending body.
        body: usize,
        /// The quantity that is not finite.
        quantity: StateQuantity,
    },
    /// A body has a negative inverse mass.
    #[error("body {body} has the negative inverse mass {value}")]
    NegativeInverseMass {
        /// Index of the offending body.
        body: usize,
        /// The negative inverse mass.
        value: f64,
    },
}
/// Records per-stage timing samples and provides basic statistics.
///
/// Each call to [`PipelineProfiler::record`] appends one sample for the
/// named stage.  [`PipelineProfiler::summary`] returns mean ± stddev.
#[derive(Debug, Default)]
pub struct PipelineProfiler {
    pub(super) samples: std::collections::HashMap<String, Vec<f64>>,
}
impl PipelineProfiler {
    /// Create a new, empty profiler.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append a timing sample (milliseconds) for `stage_name`.
    pub fn record(&mut self, stage_name: &str, ms: f64) {
        self.samples
            .entry(stage_name.to_owned())
            .or_default()
            .push(ms);
    }
    /// Returns `(mean_ms, stddev_ms, sample_count)` for `stage_name`, or
    /// `None` if no samples have been recorded.
    pub fn summary(&self, stage_name: &str) -> Option<(f64, f64, usize)> {
        let v = self.samples.get(stage_name)?;
        if v.is_empty() {
            return None;
        }
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let variance = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        Some((mean, variance.sqrt(), v.len()))
    }
    /// Return all stage names that have recorded samples.
    pub fn stage_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.samples.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
    /// Total number of samples across all stages.
    pub fn total_samples(&self) -> usize {
        self.samples.values().map(Vec::len).sum()
    }
    /// Clear all recorded samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}
/// Models multiple compute queues (graphics queue + async compute queue).
///
/// On Vulkan/Metal/DX12 certain hardware can overlap work on the graphics
/// queue and an async compute queue.  This mock records which queue each
/// dispatch was submitted to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueType {
    /// Main graphics/compute queue.
    Main,
    /// Async compute queue (can overlap with graphics).
    AsyncCompute,
    /// Transfer/copy queue.
    Transfer,
}
/// Wraps an [`Instant`] and records the elapsed duration of a stage.
pub struct StageTimer {
    pub(super) start: Instant,
    /// Elapsed time recorded by [`StageTimer::stop`].
    pub elapsed_ms: f64,
}
impl StageTimer {
    /// Create a new timer and start the clock.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            elapsed_ms: 0.0,
        }
    }
    /// Stop the timer and record elapsed time in milliseconds.
    pub fn stop(&mut self) {
        self.elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;
    }
}
/// Low-level GPU pipeline statistics for one dispatch (mock).
///
/// Mirrors the statistics provided by `VkQueryPool` with
/// `VK_QUERY_TYPE_PIPELINE_STATISTICS` or Metal/D3D equivalents.
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    /// Number of invocations of the compute shader.
    pub cs_invocations: u64,
    /// Number of work-groups dispatched.
    pub workgroups_dispatched: u64,
    /// Estimated FLOP count (mock value).
    pub flops: u64,
    /// Estimated bytes read from global memory.
    pub bytes_read: u64,
    /// Estimated bytes written to global memory.
    pub bytes_written: u64,
}
impl PipelineStatistics {
    /// Compute the arithmetic intensity (FLOPs per byte).
    pub fn arithmetic_intensity(&self) -> f64 {
        let bytes = self.bytes_read + self.bytes_written;
        if bytes == 0 {
            return 0.0;
        }
        self.flops as f64 / bytes as f64
    }
    /// Estimated memory bandwidth utilization as a fraction of `peak_bw_bytes_s`.
    pub fn bandwidth_utilization(&self, peak_bw_bytes_s: f64, elapsed_s: f64) -> f64 {
        if peak_bw_bytes_s <= 0.0 || elapsed_s <= 0.0 {
            return 0.0;
        }
        let used = (self.bytes_read + self.bytes_written) as f64 / elapsed_s;
        (used / peak_bw_bytes_s).min(1.0)
    }
}
/// Multi-queue command recorder: tracks which batches go to which queue.
#[derive(Debug, Default)]
pub struct MultiQueueRecorder {
    /// Pending batches on the main queue.
    pub main_queue: Vec<DispatchBatch>,
    /// Pending batches on the async compute queue.
    pub async_queue: Vec<DispatchBatch>,
    /// Pending batches on the transfer queue.
    pub transfer_queue: Vec<DispatchBatch>,
    /// Total batches ever recorded.
    pub total_recorded: usize,
}
impl MultiQueueRecorder {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Submit a batch to the specified queue.
    pub fn submit(&mut self, batch: DispatchBatch, queue: QueueType) {
        self.total_recorded += 1;
        match queue {
            QueueType::Main => self.main_queue.push(batch),
            QueueType::AsyncCompute => self.async_queue.push(batch),
            QueueType::Transfer => self.transfer_queue.push(batch),
        }
    }
    /// Flush all queues and return total batches executed.
    pub fn flush_all(&mut self) -> usize {
        let n = self.main_queue.len() + self.async_queue.len() + self.transfer_queue.len();
        self.main_queue.clear();
        self.async_queue.clear();
        self.transfer_queue.clear();
        n
    }
    /// Number of pending batches across all queues.
    pub fn pending_total(&self) -> usize {
        self.main_queue.len() + self.async_queue.len() + self.transfer_queue.len()
    }
}
/// A simple slab-style GPU memory pool that sub-allocates [`CpuBuffer`]s from
/// a fixed-capacity backing store.
///
/// The pool pre-allocates a large buffer and hands out non-overlapping slices
/// (as [`CpuBuffer`] views backed by offsets).  On a real GPU this avoids per-
/// allocation overhead from `vkAllocateMemory`.
///
/// This CPU-side mock tracks allocations as `(offset, size)` pairs and
/// simulates fragmentation/free-list behaviour.
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Total capacity of the pool in `f32` elements.
    pub capacity: usize,
    /// Currently allocated `f32` elements.
    pub allocated: usize,
    /// Free-list entries `(offset_in_elements, size_in_elements)`.
    pub(super) free_list: Vec<(usize, usize)>,
}
impl GpuMemoryPool {
    /// Create a pool with the given capacity in `f32` elements.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            allocated: 0,
            free_list: vec![(0, capacity)],
        }
    }
    /// Attempt to allocate `size` `f32` elements from the pool.
    ///
    /// Returns `Some((offset, size))` on success, `None` if the pool is full.
    /// Uses a first-fit strategy.
    pub fn alloc(&mut self, size: usize) -> Option<(usize, usize)> {
        for i in 0..self.free_list.len() {
            let (off, avail) = self.free_list[i];
            if avail >= size {
                let alloc_off = off;
                if avail == size {
                    self.free_list.remove(i);
                } else {
                    self.free_list[i] = (off + size, avail - size);
                }
                self.allocated += size;
                return Some((alloc_off, size));
            }
        }
        None
    }
    /// Free a previously allocated block `(offset, size)`.
    ///
    /// Returns `Err` if the block was never allocated (invalid free).
    pub fn free(&mut self, offset: usize, size: usize) -> Result<(), &'static str> {
        if offset + size > self.capacity {
            return Err("block out of bounds");
        }
        if self.allocated < size {
            return Err("double-free: allocated count underflow");
        }
        self.allocated -= size;
        self.free_list.push((offset, size));
        self.free_list.sort_by_key(|&(off, _)| off);
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for &(off, sz) in &self.free_list {
            if let Some(last) = merged.last_mut()
                && last.0 + last.1 == off
            {
                last.1 += sz;
                continue;
            }
            merged.push((off, sz));
        }
        self.free_list = merged;
        Ok(())
    }
    /// Remaining free `f32` elements.
    pub fn free_space(&self) -> usize {
        self.capacity - self.allocated
    }
    /// Returns `true` if the pool has no outstanding allocations.
    pub fn is_fully_free(&self) -> bool {
        self.allocated == 0
    }
    /// Number of fragmented free-list entries (1 = perfectly contiguous).
    pub fn fragmentation_count(&self) -> usize {
        self.free_list.len()
    }
    /// Allocate a named [`CpuBuffer`] from the pool, returning the buffer
    /// and the pool allocation handle `(offset, size)`.
    pub fn alloc_buffer(
        &mut self,
        label: &str,
        n: usize,
        usage: BufferUsage,
    ) -> Option<(CpuBuffer, (usize, usize))> {
        let handle = self.alloc(n)?;
        let buf = CpuBuffer::new_zeros(label, n, usage);
        Some((buf, handle))
    }
}
/// A set of [`TimestampQuery`]s collected during one frame / step.
#[derive(Debug, Clone, Default)]
pub struct TimestampQuerySet {
    pub(super) queries: Vec<TimestampQuery>,
}
impl TimestampQuerySet {
    /// Create an empty query set.
    pub fn new() -> Self {
        Self::default()
    }
    /// Record a timestamp query.
    pub fn record(&mut self, query: TimestampQuery) {
        self.queries.push(query);
    }
    /// Return all queries.
    pub fn queries(&self) -> &[TimestampQuery] {
        &self.queries
    }
    /// Find the query with the longest elapsed time.
    pub fn slowest_pass(&self) -> Option<&TimestampQuery> {
        self.queries.iter().max_by(|a, b| {
            a.elapsed_ms()
                .partial_cmp(&b.elapsed_ms())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
    /// Sum of all elapsed times.
    pub fn total_elapsed_ms(&self) -> f64 {
        self.queries.iter().map(|q| q.elapsed_ms()).sum()
    }
    /// Clear all recorded queries.
    pub fn clear(&mut self) {
        self.queries.clear();
    }
}
/// Per-step statistics produced by [`PhysicsPipeline::step`].
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Time spent in the broad-phase stage (ms).
    pub broadphase_ms: f64,
    /// Time spent in the narrow-phase stage (ms).
    pub narrowphase_ms: f64,
    /// Time spent in the constraint-solve stage (ms).
    pub constraint_ms: f64,
    /// Time spent in the integration stage (ms).
    pub integration_ms: f64,
    /// Time spent in the post-process stage (ms).
    pub postprocess_ms: f64,
    /// Total wall-clock time for the step (ms).
    pub total_time_ms: f64,
    /// Number of body pairs with overlapping bounding boxes found in broad-phase
    /// (pairs of two immovable bodies are not counted).
    pub collision_pairs: u32,
    /// Number of penetrating sphere contacts the narrow phase generated.
    pub contacts: u32,
    /// Number of contacts the constraint solver built a constraint for: contacts
    /// with at least one movable body and finite velocities.
    pub solved_constraints: u32,
}
impl PipelineStats {
    /// Accumulate another stats snapshot into `self` (used for sub-step sums).
    /// The counters saturate at `u32::MAX`.
    pub fn accumulate(&mut self, other: &PipelineStats) {
        self.broadphase_ms += other.broadphase_ms;
        self.narrowphase_ms += other.narrowphase_ms;
        self.constraint_ms += other.constraint_ms;
        self.integration_ms += other.integration_ms;
        self.postprocess_ms += other.postprocess_ms;
        self.total_time_ms += other.total_time_ms;
        self.collision_pairs = self.collision_pairs.saturating_add(other.collision_pairs);
        self.contacts = self.contacts.saturating_add(other.contacts);
        self.solved_constraints = self
            .solved_constraints
            .saturating_add(other.solved_constraints);
    }
    /// Sum of all per-stage times.
    pub fn stage_total_ms(&self) -> f64 {
        self.broadphase_ms
            + self.narrowphase_ms
            + self.constraint_ms
            + self.integration_ms
            + self.postprocess_ms
    }
}
/// Optimises a sequence of resource barriers by removing redundant ones.
///
/// Two barriers are redundant when a later barrier subsumes an earlier one
/// (same `src_stage → dst_stage` pair for the same resource).
#[derive(Debug, Default)]
pub struct BarrierOptimizer;
impl BarrierOptimizer {
    /// Deduplicate `barriers`: for each `(src, dst, resource)` triple,
    /// keep only the last occurrence.  Returns the optimised set.
    pub fn optimize(barriers: &[ResourceBarrier]) -> BarrierSet {
        let mut seen: std::collections::HashMap<
            (PipelineStage, PipelineStage, &str),
            &ResourceBarrier,
        > = std::collections::HashMap::new();
        for b in barriers {
            seen.insert((b.src_stage, b.dst_stage, b.resource_name.as_str()), b);
        }
        let mut out = BarrierSet::new();
        for b in seen.values() {
            out.add(ResourceBarrier::new(
                b.src_stage,
                b.dst_stage,
                &b.resource_name,
            ));
        }
        out
    }
    /// Count how many barriers would be removed from `before` to produce
    /// the optimised set.
    pub fn savings(barriers: &[ResourceBarrier]) -> usize {
        let optimized = Self::optimize(barriers);
        barriers.len().saturating_sub(optimized.len())
    }
}
/// Fluent builder for [`PhysicsPipeline`].
///
/// ```no_run
/// use oxiphysics_gpu::pipeline::{PipelineBuilder, PipelineStage};
///
/// let pipeline = PipelineBuilder::new()
///     .substeps(2)
///     .use_gpu(false)
///     .disable_stage(PipelineStage::PostProcess)
///     .build();
///
/// assert_eq!(pipeline.config.substeps, 2);
/// assert!(!pipeline.config.is_enabled(PipelineStage::PostProcess));
/// ```
pub struct PipelineBuilder {
    pub(super) config: PipelineConfig,
}
impl PipelineBuilder {
    /// Create a builder with a default config (all stages, 1 substep, CPU,
    /// inelastic contacts).
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::new(),
        }
    }
    /// Set the number of sub-steps.
    pub fn substeps(mut self, n: u32) -> Self {
        self.config.substeps = n;
        self
    }
    /// Set whether the broad phase sorts on the GPU (see
    /// [`PipelineConfig::use_gpu`]).
    pub fn use_gpu(mut self, gpu: bool) -> Self {
        self.config.use_gpu = gpu;
        self
    }
    /// Set the number of contact-solver iterations per sub-step.
    pub fn solver_iterations(mut self, iterations: u32) -> Self {
        self.config.solver_iterations = iterations;
        self
    }
    /// Set the coefficient of restitution of every contact (clamped to `[0, 1]`
    /// when used).
    pub fn restitution(mut self, restitution: f64) -> Self {
        self.config.restitution = restitution;
        self
    }
    /// Set the Baumgarte factor of the contact solver (clamped to `[0, 1]` when
    /// used).
    pub fn baumgarte(mut self, baumgarte: f64) -> Self {
        self.config.baumgarte = baumgarte;
        self
    }
    /// Add a stage to the enabled list (idempotent).
    pub fn enable_stage(mut self, stage: PipelineStage) -> Self {
        if !self.config.enabled_stages.contains(&stage) {
            self.config.enabled_stages.push(stage);
            self.config.enabled_stages.sort();
        }
        self
    }
    /// Remove a stage from the enabled list.
    pub fn disable_stage(mut self, stage: PipelineStage) -> Self {
        self.config.enabled_stages.retain(|&s| s != stage);
        self
    }
    /// Consume the builder and produce a [`PhysicsPipeline`].
    pub fn build(self) -> PhysicsPipeline {
        PhysicsPipeline::new(self.config)
    }
}
/// A collection of [`ResourceBarrier`]s that form a complete barrier schedule
/// for one pipeline configuration.
#[derive(Debug, Clone, Default)]
pub struct BarrierSet {
    pub(super) barriers: Vec<ResourceBarrier>,
}
impl BarrierSet {
    /// Create an empty barrier set.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append a barrier.
    pub fn add(&mut self, barrier: ResourceBarrier) {
        self.barriers.push(barrier);
    }
    /// Return all barriers whose `src_stage` matches `stage`.
    pub fn barriers_from(&self, stage: PipelineStage) -> Vec<&ResourceBarrier> {
        self.barriers
            .iter()
            .filter(|b| b.src_stage == stage)
            .collect()
    }
    /// Return all barriers whose `dst_stage` matches `stage`.
    pub fn barriers_to(&self, stage: PipelineStage) -> Vec<&ResourceBarrier> {
        self.barriers
            .iter()
            .filter(|b| b.dst_stage == stage)
            .collect()
    }
    /// Number of barriers registered.
    pub fn len(&self) -> usize {
        self.barriers.len()
    }
    /// Returns `true` if no barriers are registered.
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty()
    }
    /// Validate that every barrier has `src < dst` (i.e., no backwards
    /// dependencies).  Returns the list of invalid barriers.
    pub fn validate(&self) -> Vec<&ResourceBarrier> {
        self.barriers
            .iter()
            .filter(|b| !b.is_valid_order())
            .collect()
    }
}
/// A timestamp query pair `(begin, end)` for a named pass.
///
/// On a real GPU this maps to `VkQueryPool` or `wgpu::QuerySet` writes at the
/// start and end of a render/compute pass.  Here we store wall-clock
/// `f64` timestamps in milliseconds.
#[derive(Debug, Clone)]
pub struct TimestampQuery {
    /// Human-readable label for this pass.
    pub label: String,
    /// Begin timestamp in milliseconds (relative to some epoch).
    pub begin_ms: f64,
    /// End timestamp in milliseconds.
    pub end_ms: f64,
}
impl TimestampQuery {
    /// Create a new timestamp query pair.
    pub fn new(label: impl Into<String>, begin_ms: f64, end_ms: f64) -> Self {
        Self {
            label: label.into(),
            begin_ms,
            end_ms,
        }
    }
    /// Elapsed duration in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.end_ms - self.begin_ms
    }
}
/// Tracks resource aliasing: two logical resources that share the same physical
/// memory (pool allocation) at non-overlapping lifetimes.
///
/// This is important for reducing peak GPU memory usage.
#[derive(Debug, Default)]
pub struct ResourceAliasingTracker {
    /// Maps physical allocation handle `(offset, size)` to a list of logical
    /// resource names currently using it.
    pub(super) aliases: std::collections::HashMap<(usize, usize), Vec<String>>,
}
impl ResourceAliasingTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }
    /// Record that `resource_name` uses physical allocation `(offset, size)`.
    pub fn track(&mut self, resource_name: impl Into<String>, offset: usize, size: usize) {
        self.aliases
            .entry((offset, size))
            .or_default()
            .push(resource_name.into());
    }
    /// Return all resource names sharing the physical allocation `(offset, size)`.
    pub fn aliases_for(&self, offset: usize, size: usize) -> &[String] {
        self.aliases
            .get(&(offset, size))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
    /// Return `true` if two named resources share any physical allocation.
    pub fn are_aliased(&self, a: &str, b: &str) -> bool {
        for names in self.aliases.values() {
            if names.contains(&a.to_string()) && names.contains(&b.to_string()) {
                return true;
            }
        }
        false
    }
    /// Total number of physical allocations tracked.
    pub fn allocation_count(&self) -> usize {
        self.aliases.len()
    }
    /// Total number of logical resource registrations across all allocations.
    pub fn total_resource_registrations(&self) -> usize {
        self.aliases.values().map(Vec::len).sum()
    }
}
/// Simulated GPU buffer backed by CPU memory for unit testing without a GPU.
#[derive(Debug, Clone)]
pub struct CpuBuffer {
    /// Human-readable label for debugging.
    pub label: String,
    /// Buffer contents as 32-bit floats.
    pub data: Vec<f32>,
    /// Intended usage of this buffer.
    pub usage: BufferUsage,
}
impl CpuBuffer {
    /// Create a buffer pre-filled with the given data.
    pub fn new_f32(label: &str, data: Vec<f32>, usage: BufferUsage) -> Self {
        Self {
            label: label.to_owned(),
            data,
            usage,
        }
    }
    /// Create a buffer of `n` zeros.
    pub fn new_zeros(label: &str, n: usize, usage: BufferUsage) -> Self {
        Self {
            label: label.to_owned(),
            data: vec![0.0_f32; n],
            usage,
        }
    }
    /// Number of `f32` elements in the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
/// Describes how a [`CpuBuffer`] will be used by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferUsage {
    /// Read-write storage buffer.
    Storage,
    /// Uniform (read-only, small) buffer.
    Uniform,
    /// Read-only storage buffer.
    StorageReadOnly,
}
