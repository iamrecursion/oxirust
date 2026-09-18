//! # RigidBody - Trait Implementations
//!
//! This module contains trait implementations for `RigidBody`.
//!
//! ## Implemented Traits
//!
//! - `Steppable`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use oxiphysics_core::math::Vec3;
use oxiphysics_core::{Steppable, TimeStep};

use super::rigidbody_type::RigidBody;

impl Steppable for RigidBody {
    /// Advance the body by one step under standard gravity: forces, then the
    /// implicit gyroscopic term of Euler's equation
    /// ([`RigidBody::integrate_gyroscopic`]), then positions and orientation.
    ///
    /// A torque-free body that is not isotropic therefore conserves its
    /// world-frame angular momentum to first order in the step instead of
    /// keeping its world-frame angular velocity constant; isotropic bodies are
    /// unaffected.
    fn step(&mut self, time_step: &TimeStep) {
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        self.integrate_forces(time_step.dt, &gravity);
        self.integrate_gyroscopic(time_step.dt);
        self.integrate_velocity(time_step.dt);
    }
}
