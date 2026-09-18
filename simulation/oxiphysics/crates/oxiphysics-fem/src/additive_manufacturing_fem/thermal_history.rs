// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Finite-element heat conduction for the additive-manufacturing thermal history.
//!
//! [`ThermalHistoryFem`] integrates the transient heat equation
//!
//! ```text
//!     ρ c_p ∂T/∂t = ∇·(k ∇T) + Q(x, t)
//! ```
//!
//! on a mesh of linear tetrahedra, with the moving [`GoldakSource`] as the
//! volumetric source `Q`, convective loss `h (T − T∞)` through the free surface and
//! optional nodes held at a prescribed temperature.
//!
//! # Discretisation
//!
//! * **Conduction** — the Galerkin matrix `K = Σₑ k Vₑ GₑGₑᵀ`, with `Gₑ` the
//!   gradients of the four shape functions of element `e`.  Every row sums to zero,
//!   so conduction moves heat between nodes without creating or destroying any.
//! * **Heat capacity** — row-sum lumping, `Cᵢ = ρ c_p · Σₑ Vₑ/4` over the elements
//!   that contain node `i` (J K⁻¹).
//! * **Source** — the load `fᵢ = Σₑ ∫ₑ Q Nᵢ dV`, evaluated with the four-point Gauss
//!   rule of degree 2 on every tetrahedron, so a source narrower than an element
//!   still deposits its power in the right place.
//! * **Convection** — the free surface is the set of element faces that belong to a
//!   single tetrahedron; one third of each face area is lumped to each of its three
//!   nodes, giving the exposed area `Aᵢ` (m²) and the loss `h Aᵢ (Tᵢ − T∞)` (W).
//!
//! # Time integration
//!
//! Forward Euler, `Tᵢ⁺ = Tᵢ + Δt/Cᵢ · (fᵢ − (K T)ᵢ − h Aᵢ (Tᵢ − T∞))`.  The scheme is
//! stable for `Δt ≤ 2/λ_max` with `λ_max` the largest eigenvalue of
//! `C⁻¹(K + h diag(A))`, which Gershgorin's theorem bounds by
//! `g = maxᵢ (Σⱼ |Kᵢⱼ| + h Aᵢ)/Cᵢ`.  [`ThermalHistoryFem::step`] therefore splits a
//! macro time step into the smallest number of equal sub-steps that stay below half
//! of `2/g` (on a uniform one-dimensional mesh a diffusion number `α Δt/h²` of at
//! most `1/4`, half of the stability limit), so the result never blows up and an
//! arbitrarily large `dt` can be requested.

use super::{GoldakSource, T_AMBIENT, dot3, scale3, sub3};
use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Barycentric weight of the quadrature point that favours one vertex: the larger
/// coordinate of the four-point degree-2 Gauss rule, `(5 + 3√5) / 20`.
const QUADRATURE_MAJOR: f64 = 0.585_410_196_624_968_5;

/// The three equal smaller barycentric coordinates of that rule, `(5 − √5) / 20`.
const QUADRATURE_MINOR: f64 = 0.138_196_601_125_010_5;

/// Fraction of the forward-Euler stability limit `2/g` that a sub-step may use.
const STABILITY_FRACTION: f64 = 0.5;

/// Largest number of sub-steps a single call of [`ThermalHistoryFem::step`] performs.
const MAX_SUB_STEPS: usize = 1_000_000;

/// A tetrahedron whose volume is below this multiple of its cubed longest edge is
/// rejected as degenerate.
const DEGENERACY_TOLERANCE: f64 = 1.0e-12;

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Mesh-and-material operators of the semi-discrete heat equation.
#[derive(Debug, Clone)]
struct ThermalOperators {
    /// Lumped heat capacity of every node (J K⁻¹); zero for a node that belongs to
    /// no element.
    capacity: Vec<f64>,
    /// Conduction matrix (W K⁻¹).
    conduction: CsrMatrix,
    /// Lumped free-surface area of every node (m²).
    exposed_area: Vec<f64>,
    /// `Σⱼ |Kᵢⱼ|` of every row (W K⁻¹), the Gershgorin radius of the conduction.
    abs_row_sum: Vec<f64>,
    /// Volume of every element (m³).
    element_volume: Vec<f64>,
}

/// Thermal history FEM state for tracking melt pool and temperature evolution.
///
/// Build it with [`with_mesh`](Self::with_mesh) to get a model that conducts heat:
/// the temperature of a node is then coupled to its neighbours through the
/// conduction matrix, the source deposits its power over the elements, and the free
/// surface loses heat by convection.  A model built with [`new`](Self::new) knows
/// only the node positions; it has no connectivity, hence nothing to conduct heat
/// through, and [`step`](Self::step) refuses to advance it.
///
/// `positions` and `elements` are inputs of the operators assembled at
/// construction.  After editing either of them call
/// [`refresh_operators`](Self::refresh_operators).
#[derive(Debug, Clone)]
pub struct ThermalHistoryFem {
    /// Number of nodes.
    pub n_nodes: usize,
    /// Node positions (m).
    pub positions: Vec<[f64; 3]>,
    /// Current nodal temperatures (K).
    pub temperatures: Vec<f64>,
    /// Peak temperature recorded at each node (K).
    pub peak_temperatures: Vec<f64>,
    /// Goldak heat source parameters.
    pub source: GoldakSource,
    /// Liquidus temperature (K).
    pub t_liquidus: f64,
    /// Solidus temperature (K).
    pub t_solidus: f64,
    /// Thermal conductivity (W m⁻¹ K⁻¹).
    pub conductivity: f64,
    /// Specific heat capacity (J kg⁻¹ K⁻¹).
    pub specific_heat: f64,
    /// Density (kg m⁻³).
    pub density: f64,
    /// Linear tetrahedra as node indices into `positions`; empty for a model built
    /// with [`new`](Self::new).
    pub elements: Vec<[usize; 4]>,
    /// Temperature a node is held at, if any.
    prescribed: Vec<Option<f64>>,
    /// Operators derived from the mesh and the material; `None` without a mesh.
    operators: Option<ThermalOperators>,
}

impl ThermalHistoryFem {
    /// Construct a model that holds node positions and material data but has no
    /// mesh connectivity.
    ///
    /// Such a model can report the diffusivity and classify temperatures, but it
    /// cannot be stepped: use [`with_mesh`](Self::with_mesh) for that.
    pub fn new(
        positions: Vec<[f64; 3]>,
        source: GoldakSource,
        t_liquidus: f64,
        t_solidus: f64,
        conductivity: f64,
        specific_heat: f64,
        density: f64,
    ) -> Self {
        let n = positions.len();
        Self {
            n_nodes: n,
            positions,
            temperatures: vec![T_AMBIENT; n],
            peak_temperatures: vec![T_AMBIENT; n],
            source,
            t_liquidus,
            t_solidus,
            conductivity,
            specific_heat,
            density,
            elements: Vec::new(),
            prescribed: vec![None; n],
            operators: None,
        }
    }

    /// Construct a conducting model on a tetrahedral mesh.
    ///
    /// The lumped heat capacity, the conduction matrix and the free-surface areas
    /// are assembled once here.  Every node starts at the ambient temperature.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a non-finite node position, an element that
    /// references a missing node, a degenerate (flat or inverted-to-zero)
    /// tetrahedron, a face shared by more than two elements, or a conductivity,
    /// specific heat or density that is not positive and finite.
    pub fn with_mesh(
        positions: Vec<[f64; 3]>,
        elements: Vec<[usize; 4]>,
        source: GoldakSource,
        t_liquidus: f64,
        t_solidus: f64,
        conductivity: f64,
        specific_heat: f64,
        density: f64,
    ) -> Result<Self> {
        let mut model = Self::new(
            positions,
            source,
            t_liquidus,
            t_solidus,
            conductivity,
            specific_heat,
            density,
        );
        model.elements = elements;
        model.refresh_operators()?;
        Ok(model)
    }

    /// Reassemble the operators from the current `positions`, `elements` and
    /// material properties.
    ///
    /// Call this after editing the mesh or the material.  The temperature and peak
    /// vectors must already have one entry per node.
    ///
    /// # Errors
    ///
    /// The errors of [`with_mesh`](Self::with_mesh); also
    /// [`Error::InvalidInput`] when the state vectors do not match the node count.
    /// On error the previous operators are kept.
    pub fn refresh_operators(&mut self) -> Result<()> {
        let n = self.positions.len();
        if self.temperatures.len() != n || self.peak_temperatures.len() != n {
            return Err(Error::InvalidInput(format!(
                "{n} nodes but {} temperatures and {} peak temperatures",
                self.temperatures.len(),
                self.peak_temperatures.len()
            )));
        }
        let operators = assemble_operators(
            &self.positions,
            &self.elements,
            self.conductivity,
            self.specific_heat,
            self.density,
        )?;
        self.n_nodes = n;
        self.prescribed.resize(n, None);
        self.operators = Some(operators);
        Ok(())
    }

    /// Hold `node` at `temperature` (K) from now on, e.g. a baseplate at a preheat
    /// temperature.  The node takes that temperature immediately and is excluded
    /// from the time stepping.
    ///
    /// # Errors
    ///
    /// [`Error::DofOutOfRange`] for a node that does not exist and
    /// [`Error::InvalidInput`] for a non-finite temperature.
    pub fn fix_temperature(&mut self, node: usize, temperature: f64) -> Result<()> {
        if node >= self.positions.len() {
            return Err(Error::DofOutOfRange {
                dof: node,
                n: self.positions.len(),
            });
        }
        if !temperature.is_finite() {
            return Err(Error::InvalidInput(format!(
                "prescribed temperature must be finite, got {temperature}"
            )));
        }
        self.prescribed.resize(self.positions.len(), None);
        self.prescribed[node] = Some(temperature);
        if let Some(current) = self.temperatures.get_mut(node) {
            *current = temperature;
        }
        if let Some(peak) = self.peak_temperatures.get_mut(node)
            && temperature > *peak
        {
            *peak = temperature;
        }
        Ok(())
    }

    /// Largest time step (s) for which forward Euler is guaranteed to be stable at
    /// the convection coefficient `h_conv` (W m⁻² K⁻¹), `2 / g` with the Gershgorin
    /// bound `g` described in the module documentation.
    ///
    /// Returns `None` for a model without a mesh and `Some(f64::INFINITY)` when
    /// nothing couples the nodes (no conduction, no convection).
    /// [`step`](Self::step) uses sub-steps of half this size.
    pub fn stable_time_step(&self, h_conv: f64) -> Option<f64> {
        let operators = self.operators.as_ref()?;
        let rate = stability_rate(operators, h_conv);
        Some(if rate > 0.0 {
            2.0 / rate
        } else {
            f64::INFINITY
        })
    }

    /// Advance the temperature field by `dt` seconds with the heat source centred
    /// at `source_pos` (held fixed during the step) and convective loss with the
    /// coefficient `h_conv` (W m⁻² K⁻¹) to the ambient temperature.
    ///
    /// `dt` may exceed [`stable_time_step`](Self::stable_time_step): it is divided
    /// into equal explicit sub-steps that respect the stability limit.  The peak
    /// temperatures are updated after every sub-step, so a short temperature
    /// excursion inside one call is not missed.  Nodes that belong to no element
    /// keep their temperature; nodes fixed with
    /// [`fix_temperature`](Self::fix_temperature) stay at the prescribed value.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when the model has no mesh (build it with
    /// [`with_mesh`](Self::with_mesh)), when the state vectors are out of step with
    /// the operators, when `dt` is not positive and finite, `h_conv` is negative or
    /// not finite, `source_pos` is not finite, the heat source evaluates to a
    /// non-finite load (for example a zero Goldak semi-axis), or `dt` would need
    /// more than one million sub-steps.  The state is unchanged on error.
    pub fn step(&mut self, source_pos: [f64; 3], dt: f64, h_conv: f64) -> Result<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::InvalidInput(format!(
                "time step must be finite and positive, got {dt}"
            )));
        }
        if !(h_conv.is_finite() && h_conv >= 0.0) {
            return Err(Error::InvalidInput(format!(
                "convection coefficient must be finite and non-negative, got {h_conv}"
            )));
        }
        if !source_pos.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(Error::InvalidInput(
                "heat source position must be finite".to_string(),
            ));
        }
        let Some(operators) = self.operators.as_ref() else {
            return Err(Error::InvalidInput(
                "the thermal model has no mesh connectivity to conduct heat through; \
                 build it with ThermalHistoryFem::with_mesh"
                    .to_string(),
            ));
        };
        let n = operators.capacity.len();
        if self.positions.len() != n
            || self.temperatures.len() != n
            || self.peak_temperatures.len() != n
            || self.prescribed.len() != n
        {
            return Err(Error::InvalidInput(
                "positions, temperatures and operators are out of step; \
                 call refresh_operators after editing the mesh"
                    .to_string(),
            ));
        }

        let load = self.heat_load(operators, source_pos);
        if load.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidInput(
                "the heat source produced a non-finite load (check the Goldak semi-axes)"
                    .to_string(),
            ));
        }

        let rate = stability_rate(operators, h_conv);
        let sub_step_limit = if rate > 0.0 {
            2.0 * STABILITY_FRACTION / rate
        } else {
            f64::INFINITY
        };
        let sub_steps = if dt <= sub_step_limit {
            1
        } else {
            (dt / sub_step_limit).ceil() as usize
        };
        if sub_steps > MAX_SUB_STEPS {
            return Err(Error::InvalidInput(format!(
                "a time step of {dt} s needs {sub_steps} stable sub-steps (limit \
                 {MAX_SUB_STEPS}); the largest stable step is {:e} s",
                2.0 / rate
            )));
        }
        let sub_dt = dt / sub_steps as f64;

        let mut temperature = self.temperatures.clone();
        let mut peak = self.peak_temperatures.clone();
        let mut conduction_flux = vec![0.0; n];
        for _ in 0..sub_steps {
            operators
                .conduction
                .mul_vec_axpby(&temperature, &mut conduction_flux, 1.0, 0.0);
            for node in 0..n {
                let capacity = operators.capacity[node];
                if capacity <= 0.0 {
                    continue;
                }
                if let Some(fixed) = self.prescribed[node] {
                    temperature[node] = fixed;
                    continue;
                }
                let convection =
                    h_conv * operators.exposed_area[node] * (temperature[node] - T_AMBIENT);
                let heating = load[node] - conduction_flux[node] - convection;
                temperature[node] += sub_dt * heating / capacity;
            }
            for (peak_node, &value) in peak.iter_mut().zip(&temperature) {
                if value > *peak_node {
                    *peak_node = value;
                }
            }
        }
        self.temperatures = temperature;
        self.peak_temperatures = peak;
        Ok(())
    }

    /// Thermal energy stored above the ambient temperature,
    /// `Σᵢ Cᵢ (Tᵢ − T∞)` in joules; `None` for a model without a mesh.
    pub fn stored_energy(&self) -> Option<f64> {
        let operators = self.operators.as_ref()?;
        Some(
            operators
                .capacity
                .iter()
                .zip(&self.temperatures)
                .map(|(capacity, temperature)| capacity * (temperature - T_AMBIENT))
                .sum(),
        )
    }

    /// Nodal heat input of the source (W): `fᵢ = Σₑ ∫ₑ Q Nᵢ dV` with the
    /// four-point Gauss rule on every element.
    fn heat_load(&self, operators: &ThermalOperators, source_pos: [f64; 3]) -> Vec<f64> {
        let mut load = vec![0.0; self.positions.len()];
        for (tet, &volume) in self.elements.iter().zip(&operators.element_volume) {
            let corners = [
                self.positions[tet[0]],
                self.positions[tet[1]],
                self.positions[tet[2]],
                self.positions[tet[3]],
            ];
            for major in 0..4 {
                let mut weights = [QUADRATURE_MINOR; 4];
                weights[major] = QUADRATURE_MAJOR;
                let mut point = [0.0; 3];
                for (corner, &weight) in corners.iter().zip(&weights) {
                    for (coordinate, &value) in point.iter_mut().zip(corner) {
                        *coordinate += weight * value;
                    }
                }
                let share = 0.25 * volume * self.source.heat_flux(point, source_pos);
                for (&node, &weight) in tet.iter().zip(&weights) {
                    load[node] += share * weight;
                }
            }
        }
        load
    }

    /// Return indices of nodes currently in the melt pool (T ≥ T_liquidus).
    pub fn melt_pool_nodes(&self) -> Vec<usize> {
        self.temperatures
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t >= self.t_liquidus)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return fraction of nodes above solidus (partially melted + melt pool).
    pub fn mushy_zone_fraction(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }
        let mushy = self
            .temperatures
            .iter()
            .filter(|&&t| t >= self.t_solidus)
            .count();
        mushy as f64 / self.n_nodes as f64
    }

    /// Thermal diffusivity α = k / (ρ·c_p) (m² s⁻¹).
    pub fn thermal_diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }

    /// Compute temperature-dependent conductivity using linear interpolation.
    ///
    /// k(T) = k_ref * (1 + β*(T - T_ref))
    pub fn temperature_dependent_conductivity(&self, temperature: f64, beta: f64) -> f64 {
        let t_ref = 298.15;
        self.conductivity * (1.0 + beta * (temperature - t_ref))
    }
}

/// Gershgorin bound `maxᵢ (Σⱼ |Kᵢⱼ| + h Aᵢ) / Cᵢ` (s⁻¹) on the spectral radius of
/// `C⁻¹(K + h diag(A))`, taken over the nodes that carry heat capacity.
fn stability_rate(operators: &ThermalOperators, h_conv: f64) -> f64 {
    operators
        .capacity
        .iter()
        .zip(&operators.abs_row_sum)
        .zip(&operators.exposed_area)
        .filter(|&((&capacity, _), _)| capacity > 0.0)
        .map(|((&capacity, &row_sum), &area)| (row_sum + h_conv * area) / capacity)
        .fold(0.0, f64::max)
}

/// Assemble the conduction matrix, lumped capacities and exposed areas.
fn assemble_operators(
    positions: &[[f64; 3]],
    elements: &[[usize; 4]],
    conductivity: f64,
    specific_heat: f64,
    density: f64,
) -> Result<ThermalOperators> {
    for (name, value) in [
        ("thermal conductivity", conductivity),
        ("specific heat", specific_heat),
        ("density", density),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(Error::InvalidInput(format!(
                "{name} must be positive and finite, got {value}"
            )));
        }
    }
    if let Some(index) = positions
        .iter()
        .position(|p| !p.iter().all(|coordinate| coordinate.is_finite()))
    {
        return Err(Error::InvalidInput(format!(
            "node {index} has a non-finite position"
        )));
    }

    let n = positions.len();
    let volumetric_heat_capacity = density * specific_heat;
    let mut capacity = vec![0.0; n];
    let mut element_volume = Vec::with_capacity(elements.len());
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(16 * elements.len());
    let mut faces: Vec<([usize; 3], f64)> = Vec::with_capacity(4 * elements.len());

    for (index, tet) in elements.iter().enumerate() {
        if let Some(&missing) = tet.iter().find(|&&node| node >= n) {
            return Err(Error::InvalidInput(format!(
                "element {index} references node {missing} but the model has {n} nodes"
            )));
        }
        let corners = [
            positions[tet[0]],
            positions[tet[1]],
            positions[tet[2]],
            positions[tet[3]],
        ];
        let edge_a = sub3(corners[1], corners[0]);
        let edge_b = sub3(corners[2], corners[0]);
        let edge_c = sub3(corners[3], corners[0]);
        let determinant = dot3(edge_a, cross3(edge_b, edge_c));
        let longest_edge = [
            edge_a,
            edge_b,
            edge_c,
            sub3(corners[2], corners[1]),
            sub3(corners[3], corners[1]),
            sub3(corners[3], corners[2]),
        ]
        .iter()
        .map(|edge| dot3(*edge, *edge).sqrt())
        .fold(0.0_f64, f64::max);
        let well_shaped = determinant.abs() > DEGENERACY_TOLERANCE * longest_edge.powi(3);
        if !well_shaped {
            return Err(Error::InvalidInput(format!(
                "element {index} is degenerate (zero or numerically zero volume)"
            )));
        }
        let volume = determinant.abs() / 6.0;
        element_volume.push(volume);

        // Gradients of the barycentric coordinates: ∇λ₁ = (b × c)/det and so on;
        // ∇λ₀ closes the partition of unity.
        let grad_1 = scale3(cross3(edge_b, edge_c), 1.0 / determinant);
        let grad_2 = scale3(cross3(edge_c, edge_a), 1.0 / determinant);
        let grad_3 = scale3(cross3(edge_a, edge_b), 1.0 / determinant);
        let grad_0 = [
            -(grad_1[0] + grad_2[0] + grad_3[0]),
            -(grad_1[1] + grad_2[1] + grad_3[1]),
            -(grad_1[2] + grad_2[2] + grad_3[2]),
        ];
        let gradients = [grad_0, grad_1, grad_2, grad_3];
        for (&row_node, row_gradient) in tet.iter().zip(&gradients) {
            capacity[row_node] += volumetric_heat_capacity * volume / 4.0;
            for (&column_node, column_gradient) in tet.iter().zip(&gradients) {
                triplets.push((
                    row_node,
                    column_node,
                    conductivity * volume * dot3(*row_gradient, *column_gradient),
                ));
            }
        }

        for skipped in 0..4 {
            let mut face = [0_usize; 3];
            let kept = tet
                .iter()
                .enumerate()
                .filter(|&(position, _)| position != skipped)
                .map(|(_, &node)| node);
            for (slot, node) in face.iter_mut().zip(kept) {
                *slot = node;
            }
            face.sort_unstable();
            let ab = sub3(positions[face[1]], positions[face[0]]);
            let ac = sub3(positions[face[2]], positions[face[0]]);
            let cross = cross3(ab, ac);
            faces.push((face, 0.5 * dot3(cross, cross).sqrt()));
        }
    }

    // Faces that belong to a single element form the free surface.  Sorting makes
    // the accumulation order, and with it the rounding, independent of hashing.
    faces.sort_unstable_by_key(|face| face.0);
    let mut exposed_area = vec![0.0; n];
    let mut start = 0;
    while start < faces.len() {
        let mut end = start + 1;
        while end < faces.len() && faces[end].0 == faces[start].0 {
            end += 1;
        }
        match end - start {
            1 => {
                let (face, area) = faces[start];
                for node in face {
                    exposed_area[node] += area / 3.0;
                }
            }
            2 => {}
            shared => {
                return Err(Error::InvalidInput(format!(
                    "face {:?} is shared by {shared} elements; a conforming tetrahedral mesh \
                     has at most two",
                    faces[start].0
                )));
            }
        }
        start = end;
    }

    let conduction = CsrMatrix::from_triplets(n, n, &triplets);
    let abs_row_sum: Vec<f64> = (0..n)
        .map(|row| {
            conduction.values[conduction.row_ptr[row]..conduction.row_ptr[row + 1]]
                .iter()
                .map(|value| value.abs())
                .sum()
        })
        .collect();
    Ok(ThermalOperators {
        capacity,
        conduction,
        exposed_area,
        abs_row_sum,
        element_volume,
    })
}

#[cfg(test)]
mod tests;
