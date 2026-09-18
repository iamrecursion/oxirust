//! Internal helper math functions for medical visualization.

#[inline]
pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
pub(crate) fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

#[inline]
pub(crate) fn norm3f(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
pub(crate) fn normalize3f(v: [f32; 3]) -> [f32; 3] {
    let n = norm3f(v);
    if n < 1e-10 {
        [0.0; 3]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[inline]
pub(crate) fn dot3f(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Clamp `x` to `[0, 1]`; a NaN maps to `0`.
#[inline]
pub(crate) fn clamp_unit(x: f32) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

/// Parametric interval `[t_near, t_far]` over which the ray
/// `origin + t * dir` (with `t >= 0`) lies inside the closed unit cube
/// `[0, 1]^3`, found with the slab method.
///
/// Returns `None` when the ray misses the cube, when the cube lies entirely
/// behind the origin, when `dir` is the zero vector, or when an input is not
/// finite.  `dir` does not have to be normalised, but then `t` is measured in
/// units of `|dir|`.  A ray that starts inside the cube has `t_near == 0`.
pub(crate) fn ray_unit_cube_interval(origin: [f32; 3], dir: [f32; 3]) -> Option<(f32, f32)> {
    if !origin.iter().chain(dir.iter()).all(|c| c.is_finite()) || dir == [0.0; 3] {
        return None;
    }
    let mut t_near = 0.0_f32;
    let mut t_far = f32::INFINITY;
    for axis in 0..3 {
        let (o, d) = (origin[axis], dir[axis]);
        if d == 0.0 {
            // Parallel to this pair of faces: inside the slab for every t, or never.
            if !(0.0..=1.0).contains(&o) {
                return None;
            }
            continue;
        }
        let t_enter = (0.0 - o) / d;
        let t_exit = (1.0 - o) / d;
        let (lo, hi) = if t_enter <= t_exit {
            (t_enter, t_exit)
        } else {
            (t_exit, t_enter)
        };
        t_near = t_near.max(lo);
        t_far = t_far.min(hi);
        if t_near > t_far {
            return None;
        }
    }
    t_far.is_finite().then_some((t_near, t_far))
}
