// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Small fixed-size vector, matrix and quaternion helpers shared by the
//! `WasmWorld` solvers and the joint solver of
//! [`crate::constraint_bridge`].
//!
//! Vectors are `[f64; 3]`, matrices are row-major `[[f64; 3]; 3]` and
//! quaternions are `[x, y, z, w]`.

/// A 3-vector.
pub(crate) type V3 = [f64; 3];
/// A row-major 3x3 matrix.
pub(crate) type M3 = [[f64; 3]; 3];
/// A quaternion `[x, y, z, w]`.
pub(crate) type Q4 = [f64; 4];

/// The zero matrix.
pub(crate) const M3_ZERO: M3 = [[0.0; 3]; 3];
/// The identity quaternion.
pub(crate) const Q_IDENTITY: Q4 = [0.0, 0.0, 0.0, 1.0];

/// `a + b`.
#[inline]
pub(crate) fn v_add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a - b`.
#[inline]
pub(crate) fn v_sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `a * s`.
#[inline]
pub(crate) fn v_scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `a + b * s`.
#[inline]
pub(crate) fn v_madd(a: V3, b: V3, s: f64) -> V3 {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

/// `-a`.
#[inline]
pub(crate) fn v_neg(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

/// Dot product.
#[inline]
pub(crate) fn v_dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
#[inline]
pub(crate) fn v_cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length.
#[inline]
pub(crate) fn v_length(a: V3) -> f64 {
    v_dot(a, a).sqrt()
}

/// Unit vector along `a`, or `None` when `a` is (nearly) zero or not finite.
pub(crate) fn v_normalized(a: V3) -> Option<V3> {
    let len = v_length(a);
    if len.is_finite() && len > 1e-12 {
        Some(v_scale(a, 1.0 / len))
    } else {
        None
    }
}

/// Two unit vectors that complete the unit vector `n` to a right-handed
/// orthonormal basis `(t1, t2, n)`.
pub(crate) fn v_tangent_basis(n: V3) -> (V3, V3) {
    // Pick the world axis least aligned with `n` to avoid cancellation.
    let helper = if n[0].abs() < 0.57735 {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() < 0.57735 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let t1 = v_normalized(v_cross(n, helper)).unwrap_or([0.0, 0.0, 1.0]);
    let t2 = v_cross(n, t1);
    (t1, t2)
}

/// Matrix-vector product.
#[inline]
pub(crate) fn m_vec(m: &M3, v: V3) -> V3 {
    [v_dot(m[0], v), v_dot(m[1], v), v_dot(m[2], v)]
}

/// Rotation matrix of a unit quaternion (maps body vectors to world vectors).
pub(crate) fn q_to_m3(q: Q4) -> M3 {
    let [x, y, z, w] = q;
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

/// World-space tensor `R diag(d) R^T` of a body-frame diagonal tensor.
pub(crate) fn m_rotated_diag(q: Q4, d: V3) -> M3 {
    let r = q_to_m3(q);
    let mut out = M3_ZERO;
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = r[i][0] * d[0] * r[j][0] + r[i][1] * d[1] * r[j][1] + r[i][2] * d[2] * r[j][2];
        }
    }
    out
}

/// Hamilton product `a * b`.
#[inline]
pub(crate) fn q_mul(a: Q4, b: Q4) -> Q4 {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Conjugate (the inverse of a unit quaternion).
#[inline]
pub(crate) fn q_conj(q: Q4) -> Q4 {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Normalised quaternion; the identity for a degenerate input.
pub(crate) fn q_normalized(q: Q4) -> Q4 {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2.is_finite() && n2 > 1e-300 {
        let inv = 1.0 / n2.sqrt();
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    } else {
        Q_IDENTITY
    }
}

/// Rotate a vector by a unit quaternion.
#[inline]
pub(crate) fn q_rotate(q: Q4, v: V3) -> V3 {
    m_vec(&q_to_m3(q), v)
}

/// Apply a small world-space rotation vector `theta` to the orientation `q`
/// (`q' = normalize(q + 0.5 [theta, 0] q)`).
pub(crate) fn q_apply_rotation_vector(q: Q4, theta: V3) -> Q4 {
    let dq = q_mul([theta[0], theta[1], theta[2], 0.0], q);
    q_normalized([
        q[0] + 0.5 * dq[0],
        q[1] + 0.5 * dq[1],
        q[2] + 0.5 * dq[2],
        q[3] + 0.5 * dq[3],
    ])
}

/// Rotation vector (axis * angle, angle in `[0, pi]`) of a unit quaternion.
pub(crate) fn q_to_rotation_vector(q: Q4) -> V3 {
    // Use the representative with a non-negative scalar part (shortest arc).
    let q = if q[3] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    };
    let s = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
    if s < 1e-12 {
        // First-order expansion: angle * axis ~= 2 * xyz.
        return [2.0 * q[0], 2.0 * q[1], 2.0 * q[2]];
    }
    let angle = 2.0 * s.atan2(q[3]);
    v_scale([q[0], q[1], q[2]], angle / s)
}

/// Wrap an angle into `(-pi, pi]`.
pub(crate) fn wrap_angle(a: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut r = a % two_pi;
    if r <= -std::f64::consts::PI {
        r += two_pi;
    } else if r > std::f64::consts::PI {
        r -= two_pi;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_basis_is_orthonormal() {
        for n in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]] {
            let (t1, t2) = v_tangent_basis(n);
            assert!(v_dot(t1, n).abs() < 1e-12);
            assert!(v_dot(t2, n).abs() < 1e-12);
            assert!(v_dot(t1, t2).abs() < 1e-12);
            assert!((v_length(t1) - 1.0).abs() < 1e-12);
            assert!((v_length(t2) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn rotation_vector_round_trip() {
        let angle = 0.7_f64;
        let q = [0.0, (angle * 0.5).sin(), 0.0, (angle * 0.5).cos()];
        let rv = q_to_rotation_vector(q);
        assert!((rv[1] - angle).abs() < 1e-12);
        assert!(rv[0].abs() < 1e-12 && rv[2].abs() < 1e-12);
        let rotated = q_rotate(q, [1.0, 0.0, 0.0]);
        assert!((rotated[0] - angle.cos()).abs() < 1e-12);
        assert!((rotated[2] + angle.sin()).abs() < 1e-12);
    }

    #[test]
    fn rotated_diag_matches_identity_for_isotropic_tensor() {
        let q = q_normalized([0.3, -0.2, 0.5, 0.8]);
        let m = m_rotated_diag(q, [2.0, 2.0, 2.0]);
        for (i, row) in m.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!((cell - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn wrap_angle_range() {
        assert!((wrap_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-12);
        assert!((wrap_angle(-0.5) + 0.5).abs() < 1e-15);
        assert!((wrap_angle(7.0) - (7.0 - std::f64::consts::TAU)).abs() < 1e-12);
    }
}
