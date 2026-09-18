//! # SdfCone - Trait Implementations
//!
//! This module contains trait implementations for `SdfCone`.
//!
//! ## Implemented Traits
//!
//! - `ImplicitSurface`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::ImplicitSurface;
use super::functions::normalize;
use super::types::SdfCone;
use crate::implicit_geometry::functions::Sdf;
use crate::implicit_geometry::types::SdfCone as ConeProfile;

impl ImplicitSurface for SdfCone {
    /// Exact signed distance to the finite cone (negative inside).
    ///
    /// The cone is a body of revolution about the axis through `apex` along
    /// `+Y`, so the distance is computed in the meridian half-plane by
    /// [`ConeProfile`] (the minimum distance to the slant segment and to the
    /// base segment, with the sign of the enclosing region), on the coordinates
    /// relative to the apex.  Parameters outside the valid range are treated as
    /// documented there.
    fn sdf(&self, p: [f64; 3]) -> f64 {
        ConeProfile::new(self.half_angle, self.height).dist([
            p[0] - self.apex[0],
            p[1] - self.apex[1],
            p[2] - self.apex[2],
        ])
    }
    fn gradient(&self, p: [f64; 3]) -> [f64; 3] {
        const EPS: f64 = 1e-5;
        let gx = self.sdf([p[0] + EPS, p[1], p[2]]) - self.sdf([p[0] - EPS, p[1], p[2]]);
        let gy = self.sdf([p[0], p[1] + EPS, p[2]]) - self.sdf([p[0], p[1] - EPS, p[2]]);
        let gz = self.sdf([p[0], p[1], p[2] + EPS]) - self.sdf([p[0], p[1], p[2] - EPS]);
        normalize([gx, gy, gz])
    }
}
