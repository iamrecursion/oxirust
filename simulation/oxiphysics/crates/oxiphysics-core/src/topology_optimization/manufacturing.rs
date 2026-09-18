// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Manufacturing-aware filters for topology optimization.

use super::GridSize;
use std::f64::consts::FRAC_PI_2;

/// Integer exponent `P` of the p-norm that serves as the smooth maximum of
/// the support in the additive-manufacturing filter (Langelaar uses 40).
const AM_SMAX_EXPONENT: i32 = 40;

/// Regularisation `ε` of the smooth minimum of the additive-manufacturing
/// filter; the printed density may exceed the design density by `√ε / 2`.
const AM_SMIN_EPSILON: f64 = 1e-4;

/// Manufacturing-aware filters for topology optimization.
///
/// Enforces practical constraints such as minimum length scale, overhang
/// limit for additive manufacturing, and cast/mill/extrude directions.
#[derive(Debug, Clone)]
pub struct ManufacturingFilters {
    /// Grid size.
    pub grid: GridSize,
    /// Minimum member size (in element units).
    pub min_member_size: f64,
    /// Maximum overhang angle in radians (for additive manufacturing),
    /// measured from the build direction: a surface may lean away from the
    /// vertical by at most this angle without support. `π/4` is the classical
    /// 45° rule, `0` allows only vertical walls and `π/2` or more allows
    /// anything; see [`ManufacturingFilters::overhang_filter`].
    pub max_overhang_angle: f64,
    /// Build direction: 0=+y, 1=-y, 2=+x, 3=-x.
    pub build_direction: usize,
}

impl ManufacturingFilters {
    /// Creates a new `ManufacturingFilters`.
    pub fn new(grid: GridSize, min_member_size: f64, max_overhang_angle: f64) -> Self {
        Self {
            grid,
            min_member_size,
            max_overhang_angle,
            build_direction: 0,
        }
    }

    /// Sets the build direction.
    pub fn with_build_direction(mut self, dir: usize) -> Self {
        self.build_direction = dir;
        self
    }

    /// Applies minimum length scale filter by thresholding Heaviside-projected density.
    ///
    /// Projects to solid (ρ → 1) if enough neighbors are solid, else void (ρ → 0).
    pub fn minimum_length_scale(&self, density: &[f64], beta: f64) -> Vec<f64> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let r = self.min_member_size * 0.5;
        density
            .iter()
            .enumerate()
            .map(|(e, &rho)| {
                let (row, col) = self.grid.row_col(e);
                // Count neighbors within radius that are solid
                let mut count = 0;
                let mut total = 0;
                for dr in -(r as i64)..=(r as i64) {
                    for dc in -(r as i64)..=(r as i64) {
                        if (dr as f64 * dr as f64 + dc as f64 * dc as f64).sqrt() > r {
                            continue;
                        }
                        let nr = row as i64 + dr;
                        let nc = col as i64 + dc;
                        if nr < 0 || nr >= ny as i64 || nc < 0 || nc >= nx as i64 {
                            continue;
                        }
                        let ni = nr as usize * nx + nc as usize;
                        if density[ni] > 0.5 {
                            count += 1;
                        }
                        total += 1;
                    }
                }
                let frac = if total > 0 {
                    count as f64 / total as f64
                } else {
                    rho
                };
                // Smooth threshold
                1.0 / (1.0 + (-beta * (frac - 0.5)).exp())
            })
            .collect()
    }

    /// Applies the additive-manufacturing overhang filter of Langelaar
    /// (2016, 2017) and returns the *printed* density.
    ///
    /// The design is built layer by layer from the base plate along the build
    /// direction (`0` = +y, `1` = -y, `2` = +x, anything else = -x; the base
    /// plate is row 0, the last row, column 0 and the last column
    /// respectively). The first layer rests on the plate and is printed as
    /// designed. Every later element is printed only as far as the printed
    /// layer below supports it:
    ///
    /// `ρ̂_i = smin(ρ_i, ξ_i)`, `ξ_i = smax_j (w_j ρ̂_j)`,
    ///
    /// where `j` runs over the support footprint of `i` in the previous
    /// layer, `smax` is the p-norm `(Σ x^P)^(1/P)` with `P = 40` and
    /// `smin(x, y) = ½ (x + y - sqrt((x - y)² + ε) + sqrt ε)` with
    /// `ε = 1e-4`, smooth stand-ins for the maximum and the minimum. Because
    /// the sweep runs layer by layer, unsupported material is cut off from
    /// everything above it as well: a floating block is printed as (almost)
    /// void, and fully supported material keeps its density. The smoothing
    /// costs a small error: `smin(x, y)` exceeds `min(x, y)` by at most
    /// `sqrt(ε)/2 = 0.005`, so supported material prints at up to
    /// `ρ̂ ≤ ρ + 0.005` and unsupported material leaks about `0.005` of
    /// density per layer (`ρ̂ ≈ 0.005 n` in the `n`-th layer above the
    /// missing support).
    ///
    /// The footprint follows `max_overhang_angle` `θ`, measured from the
    /// build direction. An overhang may reach `tan θ` element widths beyond
    /// the material below per layer, so the footprint of an element is the
    /// `2 ⌈tan θ⌉ + 1` elements of the layer below within `⌈tan θ⌉` positions
    /// of it. For the classical `θ = π/4` these are the three elements
    /// directly below and diagonally below, for `θ = 0` only the one directly
    /// below, and `θ ≥ π/2` imposes no constraint (the input is returned).
    /// When `tan θ` is not an integer the outermost footprint elements enter
    /// with the weight `w_j = tan θ - ⌊tan θ⌋`, the share of their width that
    /// the overhang reaches; all other elements have weight one. This makes
    /// the filter vary continuously with the angle. A footprint cut off by
    /// the side of the grid simply has fewer elements.
    ///
    /// `density` must hold `grid.n_elem()` entries; any other length is
    /// returned unchanged. [`ManufacturingFilters::overhang_filter_sensitivity`]
    /// maps sensitivities back through the filter.
    ///
    /// # Examples
    ///
    /// A bar in the third row that rests on nothing prints as void, while the
    /// same bar standing on a column keeps its density:
    ///
    /// ```
    /// use oxiphysics_core::topology_optimization::{GridSize, ManufacturingFilters};
    ///
    /// let filter = ManufacturingFilters::new(GridSize::new(5, 4), 2.0, std::f64::consts::FRAC_PI_4);
    /// let mut floating = vec![0.0; 20];
    /// for col in 1..4 {
    ///     floating[2 * 5 + col] = 1.0;
    /// }
    /// assert!(filter.overhang_filter(&floating)[2 * 5 + 2] < 0.02);
    ///
    /// let mut supported = floating.clone();
    /// supported[2] = 1.0; // base plate
    /// supported[5 + 2] = 1.0;
    /// assert!(filter.overhang_filter(&supported)[2 * 5 + 2] > 0.99);
    /// ```
    pub fn overhang_filter(&self, density: &[f64]) -> Vec<f64> {
        match OverhangSweep::new(self, density) {
            Some(sweep) => sweep.printed,
            None => density.to_vec(),
        }
    }

    /// Maps the sensitivity with respect to the printed density back to the
    /// design density, the adjoint of [`ManufacturingFilters::overhang_filter`].
    ///
    /// `sensitivity[i]` is `∂F/∂ρ̂_i` for some objective or constraint `F`;
    /// the result is `∂F/∂ρ_i`. It is the exact derivative of the filter,
    /// computed by repeating the forward sweep and then running it backwards
    /// from the last layer to the base plate (reverse-mode differentiation), at
    /// the cost of one extra pass. The p-norm is not differentiable where all
    /// elements of a footprint are exactly zero; there the support receives no
    /// sensitivity.
    ///
    /// If the filter is the identity (`max_overhang_angle >= π/2`) or either
    /// slice does not have `grid.n_elem()` entries, `sensitivity` is returned
    /// unchanged.
    pub fn overhang_filter_sensitivity(&self, density: &[f64], sensitivity: &[f64]) -> Vec<f64> {
        if sensitivity.len() != density.len() {
            return sensitivity.to_vec();
        }
        let Some(sweep) = OverhangSweep::new(self, density) else {
            return sensitivity.to_vec();
        };
        let layers = sweep.layers;
        let mut adjoint = sensitivity.to_vec();
        let mut design = vec![0.0f64; density.len()];
        for layer in (0..layers.count()).rev() {
            for position in 0..layers.width() {
                let i = layers.index(layer, position);
                if layer == 0 {
                    design[i] = adjoint[i];
                    continue;
                }
                let support = sweep.support[i];
                let (d_design, d_support) = smooth_min_partials(density[i], support);
                design[i] = adjoint[i] * d_design;
                let upstream = adjoint[i] * d_support;
                if upstream == 0.0 || support <= 0.0 {
                    continue;
                }
                for (supporter, weight) in sweep.supporters(layer, position) {
                    let scaled = weight * sweep.printed[supporter].max(0.0);
                    if scaled > 0.0 {
                        adjoint[supporter] +=
                            upstream * weight * (scaled / support).powi(AM_SMAX_EXPONENT - 1);
                    }
                }
            }
        }
        design
    }

    /// Applies the casting (demoulding) constraint.
    ///
    /// A cast part must be removable from its mould by a single pull along
    /// one direction, which rules out undercuts: whenever an element has
    /// material farther out along the pull direction, every element between
    /// it and the parting surface must hold material too. The pull direction
    /// follows the `build_direction` convention (`0` = +y, `1` = -y,
    /// `2` = +x, anything else = -x), with the parting surface on the
    /// opposite side: row 0 for +y, the last row for -y, column 0 for +x and
    /// the last column for -x.
    ///
    /// Each grid line parallel to the pull direction is swept from its free
    /// end towards the parting surface keeping the running maximum density.
    /// The result is therefore monotone along the pull axis, never lower than
    /// the input (so it only adds material), stays within the range of the
    /// input densities, and is idempotent.
    ///
    /// `density` must hold `grid.n_elem()` entries; any other length is
    /// returned unchanged.
    pub fn casting_filter(&self, density: &[f64]) -> Vec<f64> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let mut result = density.to_vec();
        if nx == 0 || ny == 0 || density.len() != nx * ny {
            return result;
        }
        match self.build_direction {
            0 => {
                for row in (0..ny - 1).rev() {
                    for col in 0..nx {
                        let farther = result[(row + 1) * nx + col];
                        let cell = &mut result[row * nx + col];
                        *cell = cell.max(farther);
                    }
                }
            }
            1 => {
                for row in 0..ny - 1 {
                    for col in 0..nx {
                        let farther = result[row * nx + col];
                        let cell = &mut result[(row + 1) * nx + col];
                        *cell = cell.max(farther);
                    }
                }
            }
            2 => {
                for row in 0..ny {
                    for col in (1..nx).rev() {
                        let farther = result[row * nx + col];
                        let cell = &mut result[row * nx + col - 1];
                        *cell = cell.max(farther);
                    }
                }
            }
            _ => {
                for row in 0..ny {
                    for col in 0..nx - 1 {
                        let farther = result[row * nx + col];
                        let cell = &mut result[row * nx + col + 1];
                        *cell = cell.max(farther);
                    }
                }
            }
        }
        result
    }

    /// Applies extrusion filter (constant cross-section constraint).
    ///
    /// Each column (in build direction) is assigned the maximum density across
    /// the extrusion depth, enforcing a prismatic shape.
    pub fn extrusion_filter(&self, density: &[f64], extrude_dir: usize) -> Vec<f64> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let mut result = density.to_vec();
        if extrude_dir == 0 {
            // Extrude in x direction: each column gets max over rows
            for col in 0..nx {
                let max_val: f64 = (0..ny)
                    .map(|r| density[r * nx + col])
                    .fold(0.0f64, f64::max);
                for row in 0..ny {
                    result[row * nx + col] = max_val;
                }
            }
        } else {
            // Extrude in y direction: each row gets max over cols
            for row in 0..ny {
                let max_val: f64 = (0..nx)
                    .map(|c| density[row * nx + c])
                    .fold(0.0f64, f64::max);
                for col in 0..nx {
                    result[row * nx + col] = max_val;
                }
            }
        }
        result
    }

    /// Applies milling constraint: ensures no enclosed voids.
    ///
    /// Flood-fills from the boundary and marks all reachable voids as accessible.
    pub fn milling_filter(&self, density: &[f64], threshold: f64) -> Vec<f64> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let n = nx * ny;
        let mut accessible = vec![false; n];
        let mut queue = std::collections::VecDeque::new();

        // Seed from boundary
        for row in 0..ny {
            for col in 0..nx {
                if row == 0 || row == ny - 1 || col == 0 || col == nx - 1 {
                    let e = row * nx + col;
                    if density[e] < threshold {
                        accessible[e] = true;
                        queue.push_back(e);
                    }
                }
            }
        }

        // BFS flood fill through voids
        while let Some(e) = queue.pop_front() {
            let (row, col) = (e / nx, e % nx);
            let neighbors = [
                if row > 0 {
                    Some((row - 1) * nx + col)
                } else {
                    None
                },
                if row + 1 < ny {
                    Some((row + 1) * nx + col)
                } else {
                    None
                },
                if col > 0 {
                    Some(row * nx + col - 1)
                } else {
                    None
                },
                if col + 1 < nx {
                    Some(row * nx + col + 1)
                } else {
                    None
                },
            ];
            for nbr in neighbors.into_iter().flatten() {
                if !accessible[nbr] && density[nbr] < threshold {
                    accessible[nbr] = true;
                    queue.push_back(nbr);
                }
            }
        }

        // Fill enclosed voids with solid
        density
            .iter()
            .enumerate()
            .map(|(e, &rho)| {
                if rho < threshold && !accessible[e] {
                    1.0
                } else {
                    rho
                }
            })
            .collect()
    }
}

/// Layer-by-layer view of the grid for one build direction.
#[derive(Debug, Clone, Copy)]
struct BuildLayers {
    nx: usize,
    ny: usize,
    direction: usize,
}

impl BuildLayers {
    /// Number of layers from the base plate to the free end.
    fn count(&self) -> usize {
        if self.direction < 2 { self.ny } else { self.nx }
    }

    /// Number of elements per layer.
    fn width(&self) -> usize {
        if self.direction < 2 { self.nx } else { self.ny }
    }

    /// Row-major index of the element at `position` along `layer`, layer `0`
    /// being the one on the base plate.
    fn index(&self, layer: usize, position: usize) -> usize {
        match self.direction {
            0 => layer * self.nx + position,
            1 => (self.ny - 1 - layer) * self.nx + position,
            2 => position * self.nx + layer,
            _ => position * self.nx + (self.nx - 1 - layer),
        }
    }
}

/// Forward sweep of the additive-manufacturing filter, kept for the adjoint.
#[derive(Debug, Clone)]
struct OverhangSweep {
    layers: BuildLayers,
    /// Footprint weights for the offsets `-reach..=reach`.
    weights: Vec<f64>,
    /// Printed density `ρ̂` per element.
    printed: Vec<f64>,
    /// Support `ξ` per element (one on the base plate).
    support: Vec<f64>,
}

impl OverhangSweep {
    /// Runs the filter; `None` if it is the identity or the input has the
    /// wrong size.
    fn new(filters: &ManufacturingFilters, density: &[f64]) -> Option<Self> {
        let (nx, ny) = (filters.grid.nx, filters.grid.ny);
        let layers = BuildLayers {
            nx,
            ny,
            direction: filters.build_direction,
        };
        if nx == 0 || ny == 0 || density.len() != nx * ny {
            return None;
        }
        let weights = footprint_weights(filters.max_overhang_angle, layers.width())?;
        let mut sweep = Self {
            layers,
            weights,
            printed: vec![0.0; density.len()],
            support: vec![1.0; density.len()],
        };
        for layer in 0..layers.count() {
            for position in 0..layers.width() {
                let i = layers.index(layer, position);
                if layer == 0 {
                    sweep.printed[i] = density[i];
                    continue;
                }
                let support = sweep.support_at(layer, position);
                sweep.support[i] = support;
                sweep.printed[i] = smooth_min(density[i], support);
            }
        }
        Some(sweep)
    }

    /// Half-width of the footprint in elements.
    fn reach(&self) -> usize {
        self.weights.len() / 2
    }

    /// `(index, weight)` of the elements of `layer - 1` that support the
    /// element at `position` of `layer`.
    fn supporters(&self, layer: usize, position: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let reach = self.reach();
        let first = position.saturating_sub(reach);
        let last = (position + reach).min(self.layers.width() - 1);
        (first..=last).map(move |other| {
            let weight = self.weights[other + reach - position];
            (self.layers.index(layer - 1, other), weight)
        })
    }

    /// The smooth maximum `ξ` of the weighted printed densities in the
    /// footprint of the element at `position` of `layer`.
    fn support_at(&self, layer: usize, position: usize) -> f64 {
        let scaled = |(supporter, weight): (usize, f64)| weight * self.printed[supporter].max(0.0);
        let peak = self
            .supporters(layer, position)
            .map(scaled)
            .fold(0.0f64, f64::max);
        if peak <= 0.0 {
            return 0.0;
        }
        let power_sum: f64 = self
            .supporters(layer, position)
            .map(|entry| (scaled(entry) / peak).powi(AM_SMAX_EXPONENT))
            .sum();
        peak * power_sum.powf(1.0 / f64::from(AM_SMAX_EXPONENT))
    }
}

/// Footprint weights for the offsets `-reach..=reach` of the overhang angle
/// `angle` on layers of `width` elements; `None` when the angle imposes no
/// constraint. See [`ManufacturingFilters::overhang_filter`].
fn footprint_weights(angle: f64, width: usize) -> Option<Vec<f64>> {
    let angle = if angle.is_nan() { 0.0 } else { angle };
    if angle >= FRAC_PI_2 {
        return None;
    }
    let tangent = if angle > 0.0 { angle.tan() } else { 0.0 };
    // Snap a tangent that is an integer up to round-off, e.g. tan(π/4) = 1.
    let tangent = if (tangent - tangent.round()).abs() < 1e-9 {
        tangent.round()
    } else {
        tangent
    };
    let reach = (tangent.ceil() as usize).min(width.saturating_sub(1));
    Some(
        (0..=2 * reach)
            .map(|slot| {
                let distance = slot.abs_diff(reach);
                if distance == 0 {
                    1.0
                } else {
                    (tangent + 1.0 - distance as f64).clamp(0.0, 1.0)
                }
            })
            .collect(),
    )
}

/// Smooth minimum `½ (x + y - sqrt((x - y)² + ε) + sqrt ε)`.
fn smooth_min(x: f64, y: f64) -> f64 {
    0.5 * (x + y - ((x - y).powi(2) + AM_SMIN_EPSILON).sqrt() + AM_SMIN_EPSILON.sqrt())
}

/// Partial derivatives of [`smooth_min`] with respect to `x` and `y`.
fn smooth_min_partials(x: f64, y: f64) -> (f64, f64) {
    let ratio = (x - y) / ((x - y).powi(2) + AM_SMIN_EPSILON).sqrt();
    (0.5 * (1.0 - ratio), 0.5 * (1.0 + ratio))
}

#[cfg(test)]
mod tests;
