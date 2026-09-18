// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Velocity-level solver for a set of joint limit constraints.

use std::collections::HashMap;

use super::{JointComplianceParams, JointLimitConstraint, JointLimitWarmStart, StopType};

/// The limit of a joint that a constraint currently violates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViolatedLimit {
    /// The position is below the lower limit; the limit pushes the coordinate up.
    Lower,
    /// The position is above the upper limit; the limit pushes the coordinate down.
    Upper,
}

impl ViolatedLimit {
    /// Sign with which a positive limit impulse changes the joint velocity.
    fn direction(self) -> f64 {
        match self {
            Self::Lower => 1.0,
            Self::Upper => -1.0,
        }
    }

    /// Warm-start key of the `slot`-th constraint between a body pair.
    ///
    /// The violated side is part of the key: an impulse that pushed the joint up
    /// from its lower limit means nothing once the joint hits the upper limit.
    fn warm_key(self, slot: usize) -> usize {
        2 * slot
            + match self {
                Self::Lower => 0,
                Self::Upper => 1,
            }
    }
}

/// The limit violated by `c`, or `None` while the limits are disabled or the
/// position is inside them.
fn violated_limit(c: &JointLimitConstraint) -> Option<ViolatedLimit> {
    if !c.limits.enabled {
        None
    } else if c.position < c.limits.lower {
        Some(ViolatedLimit::Lower)
    } else if c.position > c.limits.upper {
        Some(ViolatedLimit::Upper)
    } else {
        None
    }
}

/// Smallest inverse effective mass for which a limit can still be resolved.
const MIN_LIMIT_INV_MASS: f64 = 1e-14;

/// One active limit as the velocity solver sees it.
///
/// With `s = direction`, the velocity condition of the limit is
/// `s * velocity >= bias`, and a limit impulse `lambda >= 0` changes the joint
/// velocity by `s * inv_mass * lambda`.
#[derive(Debug, Clone, Copy)]
struct LimitRow {
    /// Sign `s` of the limit (see above).
    direction: f64,
    /// Inverse effective mass `im_a + im_b` along the joint coordinate.
    inv_mass: f64,
    /// `inv_mass` plus the softness: the denominator of the impulse update.
    inv_mass_soft: f64,
    /// Softness `gamma` (constraint force mixing in impulse form): the limit is
    /// `s * velocity + gamma * lambda = bias`.
    softness: f64,
    /// Velocity the limit pushes the joint towards (position error correction).
    bias: f64,
    /// Key of this row in the warm-start cache.
    warm_key: usize,
}

/// Iterative solver for a collection of joint limit constraints.
///
/// Projected Gauss-Seidel on the velocity-level joint limit, `s * v >= bias`:
/// every iteration computes the impulse that moves each violated limit's joint
/// velocity onto its bias, accumulates it clamped to `>= 0` (a limit only
/// pushes, never pulls) and applies the change to `JointLimitConstraint::velocity`.
/// The positions are not integrated here; they stay fixed for the whole solve.
///
/// * The bias is the position error correction of the stop: for
///   [`StopType::Hard`] the solver-wide [`JointComplianceParams`] in
///   [`Self::compliance`] (`erp * penetration / dt`, softened by its `cfm`); for
///   [`StopType::Soft`] the implicit (backward Euler) spring-damper with that
///   stiffness and damping, i.e. `beta = h k / (h k + c)` and
///   `gamma = 1 / (h (h k + c))`.
/// * Warm starting re-applies last frame's accumulated impulse first. The cache
///   is keyed by body pair, the position of the constraint among those that share
///   the pair, and the violated side, and is rebuilt on every solve, so impulses
///   of constraints that are gone, inactive or on the other side are dropped
///   instead of piling up.
///
/// Constraints are independent rows (each carries its own relative velocity), so
/// a single iteration already reaches the solution; extra iterations leave it
/// unchanged and the returned impulse does not depend on their number.
#[derive(Debug, Clone)]
pub struct JointLimitSolver {
    /// Joint limit constraints registered with this solver.
    pub constraints: Vec<JointLimitConstraint>,
    /// Warm-start cache.
    pub warm_start: JointLimitWarmStart,
    /// Compliance of the [`StopType::Hard`] constraints.
    pub compliance: JointComplianceParams,
    /// Number of solver iterations per step.
    pub iterations: usize,
}

impl JointLimitSolver {
    /// Create a solver with default parameters.
    pub fn new(iterations: usize) -> Self {
        Self {
            constraints: Vec::new(),
            warm_start: JointLimitWarmStart::new(),
            compliance: JointComplianceParams::rigid(),
            iterations,
        }
    }

    /// Register a constraint.
    pub fn add_constraint(&mut self, c: JointLimitConstraint) {
        self.constraints.push(c);
    }

    /// For every constraint, its position among the constraints that share its
    /// body pair (the warm-start slot).
    fn pair_slots(&self) -> Vec<usize> {
        let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
        self.constraints
            .iter()
            .map(|c| {
                let count = seen.entry((c.body_a, c.body_b)).or_insert(0);
                let slot = *count;
                *count += 1;
                slot
            })
            .collect()
    }

    /// Apply warm starts from the previous frame to all constraints.
    ///
    /// Each constraint takes the impulse cached for its body pair, slot and
    /// violated side; a constraint that is inside its limits starts from zero.
    pub fn apply_warm_starts(&mut self) {
        let slots = self.pair_slots();
        for (c, slot) in self.constraints.iter_mut().zip(slots) {
            c.warm_impulse = match violated_limit(c) {
                Some(limit) => self
                    .warm_start
                    .retrieve(c.body_a, c.body_b, limit.warm_key(slot)),
                None => 0.0,
            };
        }
    }

    /// The velocity row of `c`, or `None` when it needs no impulse: the limit is
    /// not violated or disabled, or both bodies are static.
    fn limit_row(
        &self,
        c: &JointLimitConstraint,
        slot: usize,
        inv_masses: &[f64],
        dt: f64,
    ) -> Option<LimitRow> {
        let limit = violated_limit(c)?;
        let inv_mass_of = |body: usize| inv_masses.get(body).copied().unwrap_or(0.0);
        let inv_mass = inv_mass_of(c.body_a) + inv_mass_of(c.body_b);
        if inv_mass.is_nan() || inv_mass < MIN_LIMIT_INV_MASS {
            return None;
        }
        let penetration = c.limits.penetration(c.position);
        let (bias, inv_mass_soft) = match c.stop_type {
            StopType::Soft { stiffness, damping }
                if dt > 0.0 && dt * stiffness + damping > 1e-14 =>
            {
                // Implicit spring-damper: with h = dt, k = stiffness and c = damping
                // beta = h k / (h k + c) and gamma = 1 / (h (h k + c)).
                let hk_plus_c = dt * stiffness + damping;
                (
                    stiffness * penetration / hk_plus_c,
                    inv_mass + 1.0 / (dt * hk_plus_c),
                )
            }
            _ => (
                self.compliance.positional_bias(penetration, dt),
                self.compliance.softened_keff(inv_mass),
            ),
        };
        if inv_mass_soft.is_nan() || inv_mass_soft <= 0.0 {
            return None;
        }
        Some(LimitRow {
            direction: limit.direction(),
            inv_mass,
            inv_mass_soft,
            softness: inv_mass_soft - inv_mass,
            bias,
            warm_key: limit.warm_key(slot),
        })
    }

    /// Solve all constraints for `dt` seconds.
    ///
    /// `inv_masses[i]` is the inverse mass of body `i`; a body index outside the
    /// slice counts as static. On return `JointLimitConstraint::velocity` holds
    /// the corrected joint velocity (it must hold the velocity *before* the solve
    /// on entry) and `JointLimitConstraint::warm_impulse` the accumulated limit
    /// impulse.
    ///
    /// Returns the sum of the accumulated limit impulses of all violated limits,
    /// warm-started part included. For fixed inputs this is independent of
    /// [`Self::iterations`].
    pub fn solve(&mut self, inv_masses: &[f64], dt: f64) -> f64 {
        self.apply_warm_starts();
        let slots = self.pair_slots();
        let rows: Vec<Option<LimitRow>> = self
            .constraints
            .iter()
            .zip(&slots)
            .map(|(c, &slot)| self.limit_row(c, slot, inv_masses, dt))
            .collect();

        // Warm start: re-apply last frame's impulse to the velocities.
        for (c, row) in self.constraints.iter_mut().zip(&rows) {
            match row {
                Some(r) => {
                    let warm = if c.warm_impulse.is_finite() {
                        c.warm_impulse.max(0.0)
                    } else {
                        0.0
                    };
                    c.warm_impulse = warm;
                    c.velocity += r.direction * r.inv_mass * warm;
                }
                None => c.warm_impulse = 0.0,
            }
        }

        for _ in 0..self.iterations {
            for (c, row) in self.constraints.iter_mut().zip(&rows) {
                let Some(r) = row else {
                    continue;
                };
                let accumulated = c.warm_impulse;
                // Remaining violation of s * v + gamma * lambda = bias.
                let residual = r.bias - r.direction * c.velocity - r.softness * accumulated;
                let updated = (accumulated + residual / r.inv_mass_soft).max(0.0);
                c.velocity += r.direction * r.inv_mass * (updated - accumulated);
                c.warm_impulse = updated;
            }
        }

        // The cache holds exactly this frame's active impulses.
        self.warm_start.clear();
        let mut total = 0.0;
        for (c, row) in self.constraints.iter().zip(&rows) {
            if let Some(r) = row {
                total += c.warm_impulse;
                if c.warm_impulse > 0.0 {
                    self.warm_start
                        .store(c.body_a, c.body_b, r.warm_key, c.warm_impulse);
                }
            }
        }
        total
    }

    /// Clear all constraints (keep warm-start cache).
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }
}
