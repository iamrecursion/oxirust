//! # DistanceJoint - Trait Implementations
//!
//! This module contains trait implementations for `DistanceJoint`.
//!
//! ## Implemented Traits
//!
//! - `Constraint`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::traits::Constraint;
use oxiphysics_core::BodyHandle;
use oxiphysics_core::math::Vec3;
use oxiphysics_rigid::RigidBodySet;

use super::functions::{JOINT_BAUMGARTE, apply_pair_impulse, linear_effective_mass, read_body};
use super::types::DistanceJoint;

impl Constraint for DistanceJoint {
    fn prepare(&mut self, bodies: &RigidBodySet, _dt: f64) {
        let a = match read_body(bodies, self.body_a) {
            Some(p) => p,
            None => return,
        };
        let b = match read_body(bodies, self.body_b) {
            Some(p) => p,
            None => return,
        };
        self.r_a = a.rotation * self.local_anchor_a;
        self.r_b = b.rotation * self.local_anchor_b;
        let world_a = a.position + self.r_a;
        let world_b = b.position + self.r_b;
        let diff = world_b - world_a;
        let dist = diff.norm();
        self.constraint_dir = if dist > 1e-10 {
            diff / dist
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        self.eff_mass = linear_effective_mass(
            a.inv_mass,
            b.inv_mass,
            &a.inv_inertia,
            &b.inv_inertia,
            &self.r_a,
            &self.r_b,
            &self.constraint_dir,
        );
    }
    fn solve_velocity(&mut self, bodies: &mut RigidBodySet, _dt: f64) {
        let a = match read_body(bodies, self.body_a) {
            Some(p) => p,
            None => return,
        };
        let b = match read_body(bodies, self.body_b) {
            Some(p) => p,
            None => return,
        };
        let va = a.vel + a.ang_vel.cross(&self.r_a);
        let vb = b.vel + b.ang_vel.cross(&self.r_b);
        let rel_v = (vb - va).dot(&self.constraint_dir);
        let world_a = a.position + self.r_a;
        let world_b = b.position + self.r_b;
        let dist = (world_b - world_a).norm();
        let active = dist < self.min_distance || dist > self.max_distance;
        let target_v = 0.0_f64;
        if active && self.eff_mass > 1e-12 {
            // `apply_pair_impulse` adds `+P` to body A and `-P` to body B, so an
            // impulse `P = lambda * dir` changes the separation speed
            // `(v_B - v_A) . dir` by `-lambda / eff_mass`: the impulse that brings
            // the separation speed to `target_v` is `(rel_v - target_v) * eff_mass`.
            let lambda = (rel_v - target_v) * self.eff_mass;
            let impulse = self.constraint_dir * lambda;
            apply_pair_impulse(
                bodies,
                self.body_a,
                self.body_b,
                impulse,
                &self.r_a,
                &self.r_b,
            );
            self.accumulated_lambda += lambda;
        }
    }
    fn solve_position(&mut self, bodies: &mut RigidBodySet, _dt: f64) {
        let a = match read_body(bodies, self.body_a) {
            Some(p) => p,
            None => return,
        };
        let b = match read_body(bodies, self.body_b) {
            Some(p) => p,
            None => return,
        };
        let world_a = a.position + a.rotation * self.local_anchor_a;
        let world_b = b.position + b.rotation * self.local_anchor_b;
        let diff = world_b - world_a;
        let dist = diff.norm();
        let error;
        if dist < self.min_distance {
            error = dist - self.min_distance;
        } else if dist > self.max_distance {
            error = dist - self.max_distance;
        } else {
            return;
        }
        let dir = if dist > 1e-10 {
            diff / dist
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let correction = dir * (error * JOINT_BAUMGARTE);
        let k = a.inv_mass + b.inv_mass;
        if k > 1e-12 {
            if let Some(body) = bodies.get_mut(self.body_a) {
                body.transform.position += correction * (a.inv_mass / k);
            }
            if let Some(body) = bodies.get_mut(self.body_b) {
                body.transform.position -= correction * (b.inv_mass / k);
            }
        }
    }
    fn body_handles(&self) -> Vec<BodyHandle> {
        vec![self.body_a, self.body_b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joints::test_support::{
        anchor_relative_velocity, body_with_inertia, linear_momentum,
    };
    use oxiphysics_core::BodyHandle;
    use oxiphysics_rigid::RigidBody;

    const DT: f64 = 1.0 / 60.0;

    /// Total angular momentum about the origin: `sum(x x m v + I w)`.
    fn angular_momentum(bodies: &RigidBodySet) -> Vec3 {
        bodies
            .iter()
            .map(|(_, body)| {
                body.transform.position.cross(&(body.velocity * body.mass))
                    + body.world_inertia() * body.angular_velocity
            })
            .fold(Vec3::zeros(), |sum, l| sum + l)
    }

    fn pair(a: RigidBody, b: RigidBody) -> (RigidBodySet, BodyHandle, BodyHandle) {
        let mut bodies = RigidBodySet::new();
        let ha = bodies.insert(a);
        let hb = bodies.insert(b);
        (bodies, ha, hb)
    }

    fn velocity_of(bodies: &RigidBodySet, handle: BodyHandle) -> Vec3 {
        bodies.get(handle).expect("body exists").velocity
    }

    #[test]
    fn test_velocity_solve_cancels_the_separation_speed_of_a_stretched_rod() {
        let a = RigidBody::new(1.0);
        let mut b = RigidBody::new(3.0);
        // The rod is 1 m long, the anchors are 1.5 m apart and still separating.
        b.transform.position = Vec3::new(1.5, 0.0, 0.0);
        b.velocity = Vec3::new(2.0, 0.0, 0.0);
        let (mut bodies, ha, hb) = pair(a, b);
        let momentum = linear_momentum(&bodies);
        let mut joint = DistanceJoint::fixed_distance(ha, hb, Vec3::zeros(), Vec3::zeros(), 1.0);
        joint.prepare(&bodies, DT);
        joint.solve_velocity(&mut bodies, DT);
        // The impulse pair removes the separation speed and conserves the momentum
        // of 6 kg m/s: both bodies end up at 6 / 4 = 1.5 m/s.
        let (va, vb) = (velocity_of(&bodies, ha), velocity_of(&bodies, hb));
        assert!((va.x - 1.5).abs() < 1e-12, "va = {va:?}");
        assert!((vb.x - 1.5).abs() < 1e-12, "vb = {vb:?}");
        assert!((linear_momentum(&bodies) - momentum).norm() < 1e-12);
    }

    #[test]
    fn test_velocity_solve_cancels_the_approach_speed_of_a_compressed_rod() {
        let a = RigidBody::new(2.0);
        let mut b = RigidBody::new(2.0);
        // The rod is 1 m long, the anchors are 0.5 m apart and still approaching.
        b.transform.position = Vec3::new(0.0, 0.5, 0.0);
        b.velocity = Vec3::new(0.0, -3.0, 0.0);
        let (mut bodies, ha, hb) = pair(a, b);
        let mut joint = DistanceJoint::fixed_distance(ha, hb, Vec3::zeros(), Vec3::zeros(), 1.0);
        joint.prepare(&bodies, DT);
        joint.solve_velocity(&mut bodies, DT);
        let (va, vb) = (velocity_of(&bodies, ha), velocity_of(&bodies, hb));
        assert!((va.y + 1.5).abs() < 1e-12, "va = {va:?}");
        assert!((vb.y + 1.5).abs() < 1e-12, "vb = {vb:?}");
    }

    #[test]
    fn test_velocity_solve_with_lever_arms_stops_the_anchors_separating() {
        let mut a = body_with_inertia(1.0, Vec3::new(0.5, 0.5, 0.5));
        a.angular_velocity = Vec3::new(0.0, 0.0, 1.0);
        let mut b = body_with_inertia(2.0, Vec3::new(0.8, 0.8, 0.8));
        b.transform.position = Vec3::new(2.0, 0.0, 0.0);
        b.velocity = Vec3::new(3.0, 0.0, 0.0);
        b.angular_velocity = Vec3::new(0.0, 0.0, -2.0);
        let (mut bodies, ha, hb) = pair(a, b);
        let anchor_a = Vec3::new(0.0, 1.0, 0.0);
        let anchor_b = Vec3::new(0.0, -1.0, 0.0);
        // The anchors (0, 1, 0) and (2, -1, 0) are 2 sqrt 2 apart, the rod is 2 m long.
        let mut joint = DistanceJoint::fixed_distance(ha, hb, anchor_a, anchor_b, 2.0);
        let momentum = linear_momentum(&bodies);
        let angular = angular_momentum(&bodies);
        joint.prepare(&bodies, DT);
        let direction = Vec3::new(2.0, -2.0, 0.0).normalize();
        let before = anchor_relative_velocity(&bodies, ha, hb, anchor_a, anchor_b).dot(&direction);
        assert!(
            before.abs() > 0.5,
            "the anchors must move relative to each other"
        );
        joint.solve_velocity(&mut bodies, DT);
        let after = anchor_relative_velocity(&bodies, ha, hb, anchor_a, anchor_b).dot(&direction);
        assert!(after.abs() < 1e-12, "separation speed {after} instead of 0");
        assert!((linear_momentum(&bodies) - momentum).norm() < 1e-12);
        assert!((angular_momentum(&bodies) - angular).norm() < 1e-12);
    }

    #[test]
    fn test_velocity_solve_leaves_a_rod_inside_its_range_alone() {
        let a = RigidBody::new(1.0);
        let mut b = RigidBody::new(1.0);
        b.transform.position = Vec3::new(2.0, 0.0, 0.0);
        b.velocity = Vec3::new(4.0, 0.0, 0.0);
        let (mut bodies, ha, hb) = pair(a, b);
        let mut joint = DistanceJoint::new(ha, hb, Vec3::zeros(), Vec3::zeros(), 1.0, 3.0);
        joint.prepare(&bodies, DT);
        joint.solve_velocity(&mut bodies, DT);
        assert_eq!(velocity_of(&bodies, ha), Vec3::zeros());
        assert_eq!(velocity_of(&bodies, hb), Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(joint.accumulated_lambda, 0.0);
    }
}
