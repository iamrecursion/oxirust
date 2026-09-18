//! Regression tests for the Germano identity / Lilly least-squares pipeline of the
//! dynamic Smagorinsky models.
//!
//! The resolved stress `L_ij` is built here from the *definition* of the Germano
//! identity under the Smagorinsky closure (SGS stress at the test level minus the
//! test-filtered SGS stress at the grid level), independently of the `M_ij`
//! helpers that are under test.  With a constant `Cs^2` at both levels the exact
//! relation is `L_ij = Cs^2 * M_ij` with Lilly's `M_ij`, so a correct
//! implementation must recover `Cs^2` from `<L:M> / <M:M>`.

use super::functions::{
    germano_identity_residual, lilly_dynamic_constant, plane_averaged_dynamic_constant,
    strain_rate_magnitude,
};
use super::types::{DynamicSmagorinsky, DynamicSmagorinskyLilly};

/// Symmetric trace-free (incompressible) strain-rate tensor, row-major.
fn traceless_strain(a: f64, b: f64, d: f64, e: f64, f: f64) -> [f64; 9] {
    [a, d, e, d, b, f, e, f, -(a + b)]
}

/// Strain-rate tensors of the eight grid cells of one test-filter box.
fn box_cells() -> Vec<[f64; 9]> {
    vec![
        traceless_strain(0.80, -0.30, 0.50, -0.20, 0.10),
        traceless_strain(-0.40, 0.90, 0.20, 0.60, -0.30),
        traceless_strain(0.25, 0.35, -0.70, 0.15, 0.45),
        traceless_strain(-0.60, -0.20, 0.30, -0.50, 0.65),
        traceless_strain(0.55, -0.85, -0.25, 0.35, -0.15),
        traceless_strain(-0.15, 0.45, 0.80, -0.30, -0.55),
        traceless_strain(0.95, 0.10, -0.40, 0.05, 0.70),
        traceless_strain(-0.70, 0.60, 0.10, 0.75, -0.20),
    ]
}

/// Smagorinsky model at filter width `delta` with a prescribed constant.
fn smagorinsky_at(delta: f64, ratio: f64, cs2: f64) -> DynamicSmagorinskyLilly {
    let mut model = DynamicSmagorinskyLilly::new(delta, ratio);
    model.cs2 = cs2;
    model
}

/// Everything the dynamic procedure needs for one averaging box.
struct GermanoBox {
    /// Grid filter width.
    delta: f64,
    /// Test filter width.
    delta_test: f64,
    /// Resolved stress `L_ij` from the Germano identity under the closure.
    l: [f64; 9],
    /// Strain-rate magnitude of the test-filtered velocity.
    s_hat_mag: f64,
    /// Strain rate of the test-filtered velocity.
    s_hat: [f64; 9],
    /// Test-filtered product `<|S| S_ij>` of the grid-level fields.
    s_mag_s_filtered: [f64; 9],
}

/// Build the Germano data of a box of cells for a constant `cs2_true` at both
/// filter levels (box test filter = arithmetic mean over the cells).
fn germano_box(cells: &[[f64; 9]], delta: f64, ratio: f64, cs2_true: f64) -> GermanoBox {
    let n = cells.len() as f64;
    let grid = smagorinsky_at(delta, ratio, cs2_true);
    let test = smagorinsky_at(ratio * delta, ratio, cs2_true);

    let mut s_hat = [0.0_f64; 9];
    let mut s_mag_s_filtered = [0.0_f64; 9];
    let mut tau_filtered = [0.0_f64; 9];
    for s in cells {
        let s_mag = strain_rate_magnitude(*s);
        let nu_t = grid.eddy_viscosity(s_mag);
        for c in 0..9 {
            s_hat[c] += s[c] / n;
            s_mag_s_filtered[c] += s_mag * s[c] / n;
            tau_filtered[c] += -2.0 * nu_t * s[c] / n;
        }
    }
    let s_hat_mag = strain_rate_magnitude(s_hat);
    let nu_t_test = test.eddy_viscosity(s_hat_mag);

    let mut l = [0.0_f64; 9];
    for c in 0..9 {
        let t_ij = -2.0 * nu_t_test * s_hat[c];
        l[c] = t_ij - tau_filtered[c];
    }
    GermanoBox {
        delta,
        delta_test: ratio * delta,
        l,
        s_hat_mag,
        s_hat,
        s_mag_s_filtered,
    }
}

impl GermanoBox {
    /// `M_ij` through [`DynamicSmagorinsky::m_tensor_component`].
    fn m_tensor(&self) -> [f64; 9] {
        std::array::from_fn(|c| {
            DynamicSmagorinsky::m_tensor_component(
                self.delta_test * self.delta_test,
                self.s_hat_mag,
                self.s_hat[c],
                self.delta * self.delta,
                self.s_mag_s_filtered[c],
            )
        })
    }

    /// `(<L:M>, <M:M>)` summed over the nine tensor components.
    fn lm_mm(&self) -> (f64, f64) {
        let m = self.m_tensor();
        let lm: f64 = self.l.iter().zip(m.iter()).map(|(l, m)| l * m).sum();
        let mm: f64 = m.iter().map(|m| m * m).sum();
        (lm, mm)
    }
}

#[test]
fn forward_scatter_recovers_the_true_constant_through_every_entry_point() {
    let cs2_true = 0.02;
    let gb = germano_box(&box_cells(), 1.0, 2.0, cs2_true);
    let (lm, mm) = gb.lm_mm();
    assert!(mm > 1e-3, "test data must be non-degenerate: <M:M> = {mm}");
    assert!(
        lm > 0.0,
        "forward scatter must give <L:M> > 0 in Lilly's convention: {lm}"
    );

    let dynamic = DynamicSmagorinsky::dynamic_constant(lm, mm);
    assert!(
        (dynamic - cs2_true).abs() < 1e-12,
        "dynamic_constant = {dynamic}"
    );

    let mut lilly = DynamicSmagorinskyLilly::new(1.0, 2.0);
    lilly.update_cs2(lm, mm);
    assert!(
        (lilly.cs2 - cs2_true).abs() < 1e-12,
        "update_cs2 gave {} instead of {cs2_true}",
        lilly.cs2
    );

    let batched = lilly_dynamic_constant(&[lm], &[mm]);
    assert!(
        (batched - cs2_true).abs() < 1e-12,
        "lilly_dynamic_constant = {batched}"
    );

    let plane = plane_averaged_dynamic_constant(&[lm], &[mm]);
    assert!(
        (plane - cs2_true).abs() < 1e-12,
        "plane_averaged_dynamic_constant = {plane}"
    );

    let m = gb.m_tensor();
    let residual = germano_identity_residual(gb.l, m, cs2_true);
    let scale: f64 = gb.l.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        residual < 1e-12 * scale,
        "Germano identity residual {residual} (|L| = {scale})"
    );
}

#[test]
fn m_tensor_scalar_agrees_with_the_component_form_per_tensor_entry() {
    for ratio in [1.5_f64, 2.0, 3.0] {
        let delta = 0.7;
        let gb = germano_box(&box_cells(), delta, ratio, 0.02);
        let model = smagorinsky_at(delta, ratio, 0.02);
        let m_component = gb.m_tensor();
        for (c, m_c) in m_component.iter().enumerate() {
            let m_scalar = model.m_tensor_scalar(gb.s_hat_mag, gb.s_hat[c], gb.s_mag_s_filtered[c]);
            assert!(
                (m_scalar - m_c).abs() < 1e-12 * (1.0 + m_c.abs()),
                "ratio {ratio}, component {c}: {m_scalar} vs {m_c}"
            );
        }

        // Feeding the per-component Lilly M and the closure-built L into
        // update_cs2 must recover the true constant.
        let mut lilly = DynamicSmagorinskyLilly::new(delta, ratio);
        let lm: f64 = (0..9)
            .map(|c| {
                gb.l[c] * model.m_tensor_scalar(gb.s_hat_mag, gb.s_hat[c], gb.s_mag_s_filtered[c])
            })
            .sum();
        let mm: f64 = (0..9)
            .map(|c| {
                let m = model.m_tensor_scalar(gb.s_hat_mag, gb.s_hat[c], gb.s_mag_s_filtered[c]);
                m * m
            })
            .sum();
        lilly.update_cs2(lm, mm);
        assert!(
            (lilly.cs2 - 0.02).abs() < 1e-12,
            "ratio {ratio}: cs2 = {}",
            lilly.cs2
        );
    }
}

#[test]
fn recovered_constant_is_independent_of_velocity_amplitude_and_filter_width() {
    for amplitude in [0.1_f64, 1.0, 30.0] {
        for delta in [0.5_f64, 1.0, 4.0] {
            let cells: Vec<[f64; 9]> = box_cells()
                .iter()
                .map(|s| {
                    let mut scaled = *s;
                    scaled.iter_mut().for_each(|x| *x *= amplitude);
                    scaled
                })
                .collect();
            let gb = germano_box(&cells, delta, 2.0, 0.0173);
            let (lm, mm) = gb.lm_mm();
            let cs2 = DynamicSmagorinsky::dynamic_constant(lm, mm);
            assert!(
                (cs2 - 0.0173).abs() < 1e-11,
                "amplitude {amplitude}, delta {delta}: cs2 = {cs2}"
            );
        }
    }
}

#[test]
fn backscatter_is_clipped_to_zero() {
    let gb = germano_box(&box_cells(), 1.0, 2.0, -0.02);
    let (lm, mm) = gb.lm_mm();
    assert!(lm < 0.0, "backscatter data must give <L:M> < 0: {lm}");
    assert_eq!(DynamicSmagorinsky::dynamic_constant(lm, mm), 0.0);
    assert_eq!(lilly_dynamic_constant(&[lm], &[mm]), 0.0);
    assert_eq!(plane_averaged_dynamic_constant(&[lm], &[mm]), 0.0);
    let mut lilly = DynamicSmagorinskyLilly::new(1.0, 2.0);
    lilly.cs2 = 0.03;
    lilly.update_cs2(lm, mm);
    assert_eq!(lilly.cs2, 0.0);
}

#[test]
fn update_cs2_clips_to_the_stability_bound_but_dynamic_constant_does_not() {
    let gb = germano_box(&box_cells(), 1.0, 2.0, 0.09);
    let (lm, mm) = gb.lm_mm();
    let mut lilly = DynamicSmagorinskyLilly::new(1.0, 2.0);
    assert!(lilly.cs2_max < 0.09);
    lilly.update_cs2(lm, mm);
    assert_eq!(lilly.cs2, lilly.cs2_max);
    let unclipped = DynamicSmagorinsky::dynamic_constant(lm, mm);
    assert!(
        (unclipped - 0.09).abs() < 1e-12,
        "dynamic_constant = {unclipped}"
    );
}

#[test]
fn plane_average_is_the_pooled_least_squares_constant() {
    // Two averaging boxes with different true constants: the plane average is the
    // M:M-weighted mean of the two, i.e. (0.01 mm1 + 0.03 mm2) / (mm1 + mm2).
    let first = germano_box(&box_cells(), 1.0, 2.0, 0.01);
    let second_cells: Vec<[f64; 9]> = box_cells()
        .iter()
        .rev()
        .map(|s| {
            let mut scaled = *s;
            scaled.iter_mut().for_each(|x| *x *= 0.5);
            scaled
        })
        .collect();
    let second = germano_box(&second_cells, 1.0, 2.0, 0.03);
    let (lm1, mm1) = first.lm_mm();
    let (lm2, mm2) = second.lm_mm();

    let expected = (0.01 * mm1 + 0.03 * mm2) / (mm1 + mm2);
    let plane = plane_averaged_dynamic_constant(&[lm1, lm2], &[mm1, mm2]);
    let lilly = lilly_dynamic_constant(&[lm1, lm2], &[mm1, mm2]);
    assert!(
        (plane - expected).abs() < 1e-12,
        "plane = {plane}, expected {expected}"
    );
    assert!(
        (lilly - expected).abs() < 1e-12,
        "lilly = {lilly}, expected {expected}"
    );
    assert!(plane > 0.01 && plane < 0.03, "plane average {plane}");
}

#[test]
fn identical_filter_levels_give_a_vanishing_m_and_zero_constant() {
    // Test filter equal to the grid filter on a single cell: M_ij = 0 exactly, so the
    // procedure must return 0 instead of dividing by zero.
    let cells = vec![traceless_strain(0.8, -0.3, 0.5, -0.2, 0.1)];
    let gb = germano_box(&cells, 1.0, 1.0, 0.02);
    let (lm, mm) = gb.lm_mm();
    assert!(mm.abs() < 1e-30, "<M:M> = {mm}");
    assert_eq!(DynamicSmagorinsky::dynamic_constant(lm, mm), 0.0);
    let mut lilly = DynamicSmagorinskyLilly::new(1.0, 1.0);
    lilly.update_cs2(lm, mm);
    assert_eq!(lilly.cs2, 0.0);
}
