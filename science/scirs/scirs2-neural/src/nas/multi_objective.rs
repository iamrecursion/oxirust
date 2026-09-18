//! Multi-objective optimization for Neural Architecture Search
//!
//! This module provides multi-objective optimization capabilities for NAS,
//! allowing optimization of multiple conflicting objectives simultaneously
//! such as accuracy, latency, FLOPs, memory usage, and energy consumption.

use crate::error::{NeuralError, Result};
use crate::nas::{architecture_encoding::ArchitectureEncoding, EvaluationMetrics, SearchResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Represents an objective to optimize
#[derive(Debug, Clone)]
pub struct Objective {
    pub name: String,
    pub minimize: bool,
    pub weight: f64,
    pub target: Option<f64>,
    pub tolerance: Option<f64>,
}

impl Objective {
    pub fn new(name: &str, minimize: bool, weight: f64) -> Self {
        Self {
            name: name.to_string(),
            minimize,
            weight,
            target: None,
            tolerance: None,
        }
    }

    pub fn with_constraint(mut self, target: f64, tolerance: f64) -> Self {
        self.target = Some(target);
        self.tolerance = Some(tolerance);
        self
    }
}

pub struct MultiObjectiveConfig {
    pub objectives: Vec<Objective>,
    pub algorithm: MultiObjectiveAlgorithm,
    pub population_size: usize,
    pub max_generations: usize,
    pub pareto_front_limit: usize,
    pub reference_point: Option<Vec<f64>>,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            objectives: vec![
                Objective::new("validation_accuracy", false, 0.4),
                Objective::new("model_flops", true, 0.3),
                Objective::new("model_params", true, 0.2),
                Objective::new("inference_latency", true, 0.1),
            ],
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            population_size: 50,
            max_generations: 100,
            pareto_front_limit: 20,
            reference_point: None,
        }
    }
}

pub enum MultiObjectiveAlgorithm {
    NSGA2,
    SPEA2,
    MOEAD,
    HYPERE,
    WeightedSum,
    ConstraintHandling,
}

pub struct MultiObjectiveSolution {
    pub architecture: Arc<dyn ArchitectureEncoding>,
    pub objectives: Vec<f64>,
    pub constraint_violations: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
    pub dominance_count: usize,
    pub dominated_solutions: Vec<usize>,
}

impl Clone for MultiObjectiveSolution {
    fn clone(&self) -> Self {
        Self {
            architecture: self.architecture.clone(),
            objectives: self.objectives.clone(),
            constraint_violations: self.constraint_violations.clone(),
            rank: self.rank,
            crowding_distance: self.crowding_distance,
            dominance_count: self.dominance_count,
            dominated_solutions: self.dominated_solutions.clone(),
        }
    }
}

impl MultiObjectiveSolution {
    pub fn new(architecture: Arc<dyn ArchitectureEncoding>, objectives: Vec<f64>) -> Self {
        Self {
            architecture,
            objectives,
            constraint_violations: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            dominance_count: 0,
            dominated_solutions: Vec::new(),
        }
    }

    pub fn dominates(&self, other: &Self, config: &MultiObjectiveConfig) -> bool {
        let mut better = false;
        for (i, obj) in config.objectives.iter().enumerate() {
            if i >= self.objectives.len() || i >= other.objectives.len() {
                continue;
            }
            let sv = self.objectives[i];
            let ov = other.objectives[i];
            if obj.minimize {
                if sv > ov {
                    return false;
                } else if sv < ov {
                    better = true;
                }
            } else {
                if sv < ov {
                    return false;
                } else if sv > ov {
                    better = true;
                }
            }
        }
        better
    }
}

pub struct MultiObjectiveOptimizer {
    config: MultiObjectiveConfig,
    population: Vec<MultiObjectiveSolution>,
    pareto_front: Vec<MultiObjectiveSolution>,
    generation: usize,
    hypervolume_history: Vec<f64>,
}

impl MultiObjectiveOptimizer {
    pub fn new(config: MultiObjectiveConfig) -> Self {
        Self {
            config,
            population: Vec::new(),
            pareto_front: Vec::new(),
            generation: 0,
            hypervolume_history: Vec::new(),
        }
    }

    pub fn initialize_population(&mut self, results: &[SearchResult]) -> Result<()> {
        self.population.clear();
        for result in results.iter().take(self.config.population_size) {
            let objectives = self.extract_objectives(&result.metrics)?;
            self.population.push(MultiObjectiveSolution::new(
                result.architecture.clone(),
                objectives,
            ));
        }
        while self.population.len() < self.config.population_size {
            let arch = self.generate_random_architecture()?;
            let objs = self.estimate_random_objectives();
            self.population
                .push(MultiObjectiveSolution::new(arch, objs));
        }
        Ok(())
    }

    pub fn evolve_generation(&mut self) -> Result<()> {
        match self.config.algorithm {
            MultiObjectiveAlgorithm::NSGA2 => self.nsga2_step()?,
            MultiObjectiveAlgorithm::SPEA2 => self.spea2_step()?,
            MultiObjectiveAlgorithm::MOEAD => self.moead_step()?,
            MultiObjectiveAlgorithm::HYPERE => self.hypere_step()?,
            MultiObjectiveAlgorithm::WeightedSum => self.weighted_sum_step()?,
            MultiObjectiveAlgorithm::ConstraintHandling => self.constraint_handling_step()?,
        }
        self.generation += 1;
        self.update_pareto_front()?;
        let hv = self.compute_hypervolume()?;
        self.hypervolume_history.push(hv);
        Ok(())
    }

    fn nsga2_step(&mut self) -> Result<()> {
        let offspring = self.create_offspring()?;
        let mut combined = self.population.clone();
        combined.extend(offspring);
        self.non_dominated_sort(&mut combined)?;
        self.population = self.environmental_selection(combined)?;
        Ok(())
    }

    fn spea2_step(&mut self) -> Result<()> {
        let offspring = self.create_offspring()?;
        let mut combined = self.population.clone();
        combined.extend(offspring);
        self.calculate_spea2_fitness_for_population(&mut combined)?;
        self.population = self.spea2_environmental_selection(combined)?;
        Ok(())
    }

    fn moead_step(&mut self) -> Result<()> {
        let weight_vectors = self.generate_weight_vectors()?;
        for (i, weights) in weight_vectors
            .iter()
            .enumerate()
            .take(weight_vectors.len().min(self.population.len()))
        {
            let weights = weights.clone();
            let new_solution = self.update_subproblem(i, &weights)?;
            self.update_neighbors(i, &new_solution)?;
        }
        Ok(())
    }

    fn hypere_step(&mut self) -> Result<()> {
        let parent_count = 10.min(self.population.len());
        let mut offspring = Vec::new();
        for idx in 0..parent_count {
            let child_arch_box = self.population[idx].architecture.mutate(0.1)?;
            let child_arch: std::sync::Arc<
                dyn crate::nas::architecture_encoding::ArchitectureEncoding,
            > = std::sync::Arc::from(child_arch_box);
            let objectives = self.estimate_objectives(&child_arch)?;
            offspring.push(MultiObjectiveSolution::new(child_arch, objectives));
        }
        let mut combined = self.population.clone();
        combined.extend(offspring);
        self.population = self.hypervolume_environmental_selection(combined)?;
        Ok(())
    }

    fn weighted_sum_step(&mut self) -> Result<()> {
        for solution in &mut self.population {
            let ws: f64 = solution
                .objectives
                .iter()
                .zip(self.config.objectives.iter())
                .map(|(v, o)| v * o.weight)
                .sum();
            solution.objectives = vec![ws];
        }
        self.population.sort_by(|a, b| {
            let ao = a.objectives.first().copied().unwrap_or(0.0);
            let bo = b.objectives.first().copied().unwrap_or(0.0);
            ao.partial_cmp(&bo).unwrap_or(std::cmp::Ordering::Equal)
        });
        let offspring = self.create_offspring()?;
        self.population.extend(offspring);
        self.population.truncate(self.config.population_size);
        Ok(())
    }

    fn constraint_handling_step(&mut self) -> Result<()> {
        let violations: Vec<Vec<f64>> = self
            .population
            .iter()
            .map(|s| self.evaluate_constraints(s))
            .collect::<Result<Vec<_>>>()?;
        for (sol, viols) in self.population.iter_mut().zip(violations) {
            sol.constraint_violations = viols;
        }
        self.population.sort_by(|a, b| {
            let av: f64 = a.constraint_violations.iter().sum();
            let bv: f64 = b.constraint_violations.iter().sum();
            if (av - bv).abs() > 1e-12 {
                av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.objectives
                    .first()
                    .copied()
                    .unwrap_or(0.0)
                    .partial_cmp(&b.objectives.first().copied().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        let offspring = self.create_offspring()?;
        self.population = self.constraint_environmental_selection(offspring)?;
        Ok(())
    }

    fn non_dominated_sort(&self, population: &mut [MultiObjectiveSolution]) -> Result<()> {
        let n = population.len();
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut dom_counts: Vec<usize> = vec![0; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if self.dominates_by_values(&population[i].objectives, &population[j].objectives) {
                    dominated_by[i].push(j);
                } else if self
                    .dominates_by_values(&population[j].objectives, &population[i].objectives)
                {
                    dom_counts[i] += 1;
                }
            }
        }

        let mut first_front = Vec::new();
        for i in 0..n {
            population[i].dominated_solutions = dominated_by[i].clone();
            population[i].dominance_count = dom_counts[i];
            if dom_counts[i] == 0 {
                population[i].rank = 0;
                first_front.push(i);
            }
        }

        let mut fronts = vec![first_front];
        let mut fi = 0;
        while fi < fronts.len() && !fronts[fi].is_empty() {
            let mut next_front = Vec::new();
            let current = fronts[fi].clone();
            for &i in &current {
                let doms = population[i].dominated_solutions.clone();
                for &j in &doms {
                    if population[j].dominance_count > 0 {
                        population[j].dominance_count -= 1;
                        if population[j].dominance_count == 0 {
                            population[j].rank = fi + 1;
                            next_front.push(j);
                        }
                    }
                }
            }
            fi += 1;
            fronts.push(next_front);
        }
        Ok(())
    }

    fn dominates_by_values(&self, a: &[f64], b: &[f64]) -> bool {
        let mut better = false;
        for (k, obj_cfg) in self.config.objectives.iter().enumerate() {
            let oa = a.get(k).copied().unwrap_or(0.0);
            let ob = b.get(k).copied().unwrap_or(0.0);
            if obj_cfg.minimize {
                if oa > ob {
                    return false;
                } else if oa < ob {
                    better = true;
                }
            } else {
                if oa < ob {
                    return false;
                } else if oa > ob {
                    better = true;
                }
            }
        }
        better
    }

    fn calculate_crowding_distance(
        &self,
        front: &[usize],
        population: &mut [MultiObjectiveSolution],
    ) -> Result<()> {
        if front.len() <= 2 {
            for &i in front {
                population[i].crowding_distance = f64::INFINITY;
            }
            return Ok(());
        }
        for &i in front {
            population[i].crowding_distance = 0.0;
        }
        for obj_idx in 0..self.config.objectives.len() {
            let mut sorted = front.to_vec();
            sorted.sort_by(|&a, &b| {
                let oa = population[a]
                    .objectives
                    .get(obj_idx)
                    .copied()
                    .unwrap_or(0.0);
                let ob = population[b]
                    .objectives
                    .get(obj_idx)
                    .copied()
                    .unwrap_or(0.0);
                oa.partial_cmp(&ob).unwrap_or(std::cmp::Ordering::Equal)
            });
            let first = sorted[0];
            let last = sorted[sorted.len() - 1];
            population[first].crowding_distance = f64::INFINITY;
            population[last].crowding_distance = f64::INFINITY;
            let obj_min = population[first]
                .objectives
                .get(obj_idx)
                .copied()
                .unwrap_or(0.0);
            let obj_max = population[last]
                .objectives
                .get(obj_idx)
                .copied()
                .unwrap_or(0.0);
            let range = obj_max - obj_min;
            if range > 0.0 {
                for i in 1..sorted.len() - 1 {
                    let prev = population[sorted[i - 1]]
                        .objectives
                        .get(obj_idx)
                        .copied()
                        .unwrap_or(0.0);
                    let next = population[sorted[i + 1]]
                        .objectives
                        .get(obj_idx)
                        .copied()
                        .unwrap_or(0.0);
                    population[sorted[i]].crowding_distance += (next - prev) / range;
                }
            }
        }
        Ok(())
    }

    fn environmental_selection(
        &mut self,
        mut population: Vec<MultiObjectiveSolution>,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        let mut result = Vec::new();
        let mut fronts: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, s) in population.iter().enumerate() {
            fronts.entry(s.rank).or_default().push(i);
        }
        let mut current_front = 0;
        while current_front < fronts.len() {
            if let Some(front) = fronts.get(&current_front) {
                if result.len() + front.len() <= self.config.population_size {
                    for &i in front {
                        result.push(population[i].clone());
                    }
                } else {
                    self.calculate_crowding_distance(front, &mut population)?;
                    let mut fd: Vec<(usize, f64)> = front
                        .iter()
                        .map(|&i| (i, population[i].crowding_distance))
                        .collect();
                    fd.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let remaining = self.config.population_size - result.len();
                    for item in fd.into_iter().take(remaining) {
                        result.push(population[item.0].clone());
                    }
                    break;
                }
            }
            current_front += 1;
        }
        Ok(result)
    }

    fn create_offspring(&self) -> Result<Vec<MultiObjectiveSolution>> {
        if self.population.is_empty() {
            return Ok(Vec::new());
        }
        let mut offspring = Vec::new();
        for _ in 0..self.config.population_size {
            let p1 = self.tournament_selection()?;
            let p2 = self.tournament_selection()?;
            let child = p1.architecture.crossover(p2.architecture.as_ref())?;
            let mutated_box = child.mutate(0.1)?;
            let mutated: Arc<dyn ArchitectureEncoding> = Arc::from(mutated_box);
            let objectives = self.estimate_objectives(&mutated)?;
            offspring.push(MultiObjectiveSolution::new(mutated, objectives));
        }
        Ok(offspring)
    }

    fn tournament_selection(&self) -> Result<&MultiObjectiveSolution> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        if self.population.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "Population is empty".to_string(),
            ));
        }
        let mut best = rng_inst.random_range(0..self.population.len());
        for _ in 1..3 {
            let candidate = rng_inst.random_range(0..self.population.len());
            if self.is_better(&self.population[candidate], &self.population[best]) {
                best = candidate;
            }
        }
        Ok(&self.population[best])
    }

    fn is_better(&self, a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> bool {
        if a.rank < b.rank {
            true
        } else if a.rank > b.rank {
            false
        } else {
            a.crowding_distance > b.crowding_distance
        }
    }

    fn extract_objectives(&self, metrics: &EvaluationMetrics) -> Result<Vec<f64>> {
        Ok(self
            .config
            .objectives
            .iter()
            .map(|o| metrics.get(&o.name).copied().unwrap_or(0.0))
            .collect())
    }

    fn estimate_objectives(&self, _arch: &Arc<dyn ArchitectureEncoding>) -> Result<Vec<f64>> {
        Ok(self
            .config
            .objectives
            .iter()
            .map(|o| match o.name.as_str() {
                "validation_accuracy" => 0.7 + 0.2 * scirs2_core::random::random::<f64>(),
                "model_flops" => 1e6 + 1e6 * scirs2_core::random::random::<f64>(),
                "model_params" => 1e5 + 1e5 * scirs2_core::random::random::<f64>(),
                "inference_latency" => 10.0 + 10.0 * scirs2_core::random::random::<f64>(),
                _ => 0.5,
            })
            .collect())
    }

    fn generate_random_architecture(&self) -> Result<Arc<dyn ArchitectureEncoding>> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        let enc = crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng_inst)?;
        Ok(Arc::new(enc) as Arc<dyn ArchitectureEncoding>)
    }

    fn estimate_random_objectives(&self) -> Vec<f64> {
        self.config
            .objectives
            .iter()
            .map(|o| match o.name.as_str() {
                "validation_accuracy" => 0.3 + 0.4 * scirs2_core::random::random::<f64>(),
                "model_flops" => 1e5 + 1e6 * scirs2_core::random::random::<f64>(),
                "model_params" => 1e4 + 1e5 * scirs2_core::random::random::<f64>(),
                "inference_latency" => 1.0 + 20.0 * scirs2_core::random::random::<f64>(),
                _ => scirs2_core::random::random::<f64>(),
            })
            .collect()
    }

    fn update_pareto_front(&mut self) -> Result<()> {
        let mut pareto_indices = Vec::new();
        for i in 0..self.population.len() {
            let mut dominated = false;
            for j in 0..self.population.len() {
                if i != j
                    && self.dominates_by_values(
                        &self.population[j].objectives.clone(),
                        &self.population[i].objectives.clone(),
                    )
                {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                pareto_indices.push(i);
            }
        }
        let mut pareto: Vec<MultiObjectiveSolution> = pareto_indices
            .iter()
            .map(|&i| self.population[i].clone())
            .collect();
        if pareto.len() > self.config.pareto_front_limit {
            let indices: Vec<usize> = (0..pareto.len()).collect();
            self.calculate_crowding_distance(&indices, &mut pareto)?;
            pareto.sort_by(|a, b| {
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            pareto.truncate(self.config.pareto_front_limit);
        }
        self.pareto_front = pareto;
        Ok(())
    }

    fn compute_hypervolume(&self) -> Result<f64> {
        if self.pareto_front.is_empty() {
            return Ok(0.0);
        }
        let rp = self
            .config
            .reference_point
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.estimate_reference_point());
        match self.config.objectives.len() {
            2 => self.compute_hypervolume_2d(&rp),
            3 => self.compute_hypervolume_3d(&rp),
            _ => self.compute_hypervolume_monte_carlo(&rp),
        }
    }

    fn estimate_reference_point(&self) -> Vec<f64> {
        let n = self.config.objectives.len();
        let mut rp = vec![0.0f64; n];
        for (i, obj) in self.config.objectives.iter().enumerate() {
            if obj.minimize {
                let max_val = self
                    .pareto_front
                    .iter()
                    .filter_map(|s| s.objectives.get(i).copied())
                    .fold(f64::NEG_INFINITY, f64::max);
                rp[i] = if max_val.is_finite() {
                    max_val * 1.1
                } else {
                    1.0
                };
            } else {
                let min_val = self
                    .pareto_front
                    .iter()
                    .filter_map(|s| s.objectives.get(i).copied())
                    .fold(f64::INFINITY, f64::min);
                rp[i] = if min_val.is_finite() {
                    min_val * 0.9
                } else {
                    0.0
                };
            }
        }
        rp
    }

    fn compute_hypervolume_2d(&self, rp: &[f64]) -> Result<f64> {
        let min0 = self
            .config
            .objectives
            .first()
            .map(|o| o.minimize)
            .unwrap_or(true);
        let min1 = self
            .config
            .objectives
            .get(1)
            .map(|o| o.minimize)
            .unwrap_or(true);
        let rp0 = rp.first().copied().unwrap_or(0.0);
        let rp1 = rp.get(1).copied().unwrap_or(0.0);
        let mut points: Vec<(f64, f64)> = self
            .pareto_front
            .iter()
            .map(|s| {
                let v0 = s.objectives.first().copied().unwrap_or(0.0);
                let v1 = s.objectives.get(1).copied().unwrap_or(0.0);
                let x = if min0 {
                    (rp0 - v0).max(0.0)
                } else {
                    (v0 - rp0).max(0.0)
                };
                let y = if min1 {
                    (rp1 - v1).max(0.0)
                } else {
                    (v1 - rp1).max(0.0)
                };
                (x, y)
            })
            .collect();
        points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut volume = 0.0f64;
        let mut prev_y = 0.0f64;
        for (x, y) in points {
            if y > prev_y {
                volume += x * (y - prev_y);
                prev_y = y;
            }
        }
        Ok(volume)
    }

    fn compute_hypervolume_3d(&self, rp: &[f64]) -> Result<f64> {
        let points: Vec<[f64; 3]> = self
            .pareto_front
            .iter()
            .map(|s| {
                let mut arr = [0.0f64; 3];
                for (i, cell) in arr.iter_mut().enumerate() {
                    let r = rp.get(i).copied().unwrap_or(0.0);
                    let v = s.objectives.get(i).copied().unwrap_or(0.0);
                    *cell = if self
                        .config
                        .objectives
                        .get(i)
                        .map(|o| o.minimize)
                        .unwrap_or(true)
                    {
                        (r - v).max(0.0)
                    } else {
                        (v - r).max(0.0)
                    };
                }
                arr
            })
            .collect();
        let n = points.len();
        let mut volume = 0.0f64;
        for p in &points {
            volume += p[0] * p[1] * p[2];
        }
        for i in 0..n {
            for j in (i + 1)..n {
                volume -= points[i][0].min(points[j][0])
                    * points[i][1].min(points[j][1])
                    * points[i][2].min(points[j][2]);
            }
        }
        Ok(volume.max(0.0))
    }

    fn compute_hypervolume_monte_carlo(&self, rp: &[f64]) -> Result<f64> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        let num_samples = 10000usize;
        let n_obj = self.config.objectives.len();
        let mut lower_bounds = vec![f64::INFINITY; n_obj];
        let upper_bounds = rp.to_vec();
        for sol in &self.pareto_front {
            for (i, &v) in sol.objectives.iter().enumerate() {
                if i < n_obj {
                    lower_bounds[i] = lower_bounds[i].min(v);
                }
            }
        }
        for (i, lb) in lower_bounds.iter_mut().enumerate() {
            if !lb.is_finite() {
                *lb = upper_bounds.get(i).copied().unwrap_or(0.0) - 1.0;
            }
        }
        let mut dominated_count = 0usize;
        for _ in 0..num_samples {
            let sample: Vec<f64> = (0..n_obj)
                .map(|i| {
                    let lo = lower_bounds[i];
                    let hi = upper_bounds.get(i).copied().unwrap_or(lo + 1.0);
                    if hi > lo {
                        lo + rng_inst.random::<f64>() * (hi - lo)
                    } else {
                        lo
                    }
                })
                .collect();
            let mut is_dominated = false;
            'outer: for sol in &self.pareto_front {
                let mut dom = true;
                let mut better = false;
                for (i, (&sv, &pv)) in sol.objectives.iter().zip(sample.iter()).enumerate() {
                    let min = self
                        .config
                        .objectives
                        .get(i)
                        .map(|o| o.minimize)
                        .unwrap_or(true);
                    if min {
                        if sv > pv {
                            dom = false;
                            break;
                        } else if sv < pv {
                            better = true;
                        }
                    } else {
                        if sv < pv {
                            dom = false;
                            break;
                        } else if sv > pv {
                            better = true;
                        }
                    }
                }
                if dom && better {
                    is_dominated = true;
                    break 'outer;
                }
            }
            if is_dominated {
                dominated_count += 1;
            }
        }
        let sampling_vol: f64 = upper_bounds
            .iter()
            .zip(lower_bounds.iter())
            .map(|(u, l)| (u - l).max(0.0))
            .product();
        Ok(sampling_vol * (dominated_count as f64 / num_samples as f64))
    }

    pub fn get_pareto_front(&self) -> &[MultiObjectiveSolution] {
        &self.pareto_front
    }
    pub fn get_hypervolume_history(&self) -> &[f64] {
        &self.hypervolume_history
    }
    pub fn get_generation(&self) -> usize {
        self.generation
    }

    fn calculate_spea2_fitness_for_population(
        &self,
        population: &mut [MultiObjectiveSolution],
    ) -> Result<()> {
        let n = population.len();
        let mut strengths = vec![0usize; n];
        let mut raw_fitness = vec![0.0f64; n];
        let mut densities = vec![0.0f64; n];
        for i in 0..n {
            let mut count = 0;
            for j in 0..n {
                if i != j {
                    let oi: f64 = population[i].objectives.iter().sum();
                    let oj: f64 = population[j].objectives.iter().sum();
                    if oi < oj {
                        count += 1;
                    }
                }
            }
            strengths[i] = count;
        }
        for i in 0..n {
            let mut fitness = 0.0;
            for j in 0..n {
                if i != j {
                    let oi: f64 = population[i].objectives.iter().sum();
                    let oj: f64 = population[j].objectives.iter().sum();
                    if oj < oi {
                        fitness += strengths[j] as f64;
                    }
                }
            }
            raw_fitness[i] = fitness;
        }
        let k = (n as f64).sqrt() as usize;
        for i in 0..n {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| self.euclidean_distance(&population[i], &population[j]))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let kth = if k > 0 && k <= dists.len() {
                dists[k - 1]
            } else {
                dists.last().copied().unwrap_or(0.0)
            };
            densities[i] = 1.0 / (kth + 2.0);
        }
        for i in 0..n {
            population[i].crowding_distance = raw_fitness[i] + densities[i];
        }
        Ok(())
    }

    fn spea2_environmental_selection(
        &self,
        mut population: Vec<MultiObjectiveSolution>,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        population.sort_by(|a, b| {
            a.crowding_distance
                .partial_cmp(&b.crowding_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected = Vec::new();
        for sol in &population {
            if sol.crowding_distance < 1.0 && selected.len() < self.config.population_size {
                selected.push(sol.clone());
            }
        }
        if selected.len() < self.config.population_size {
            for sol in &population {
                if sol.crowding_distance >= 1.0 && selected.len() < self.config.population_size {
                    selected.push(sol.clone());
                }
            }
        }
        selected.truncate(self.config.population_size);
        Ok(selected)
    }

    fn euclidean_distance(&self, a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> f64 {
        a.objectives
            .iter()
            .zip(b.objectives.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn generate_weight_vectors(&self) -> Result<Vec<Vec<f64>>> {
        let n_obj = self.config.objectives.len();
        let n_weights = self.config.population_size;
        let mut weights = Vec::new();
        if n_obj == 2 {
            for i in 0..n_weights {
                let w1 = i as f64 / (n_weights - 1).max(1) as f64;
                weights.push(vec![w1, 1.0 - w1]);
            }
        } else {
            while weights.len() < n_weights {
                let raw: Vec<f64> = (0..n_obj)
                    .map(|_| scirs2_core::random::random::<f64>())
                    .collect();
                let sum: f64 = raw.iter().sum();
                if sum > 1e-12 {
                    weights.push(raw.iter().map(|w| w / sum).collect());
                }
            }
        }
        weights.truncate(n_weights);
        Ok(weights)
    }

    fn update_subproblem(&self, index: usize, weights: &[f64]) -> Result<MultiObjectiveSolution> {
        if index >= self.population.len() {
            return Err(NeuralError::InvalidArgument(
                "Subproblem index out of bounds".to_string(),
            ));
        }
        let current = &self.population[index];
        let neighbor = self.select_neighbor(index)?;
        let p2 = &self.population[neighbor];
        let child = current.architecture.crossover(p2.architecture.as_ref())?;
        let mutated_box = child.mutate(0.1)?;
        let mutated: Arc<dyn ArchitectureEncoding> = Arc::from(mutated_box);
        let objectives = self.estimate_objectives(&mutated)?;
        let mut child_sol = MultiObjectiveSolution::new(mutated, objectives);
        let cur_fit = self.tchebycheff_fitness(&current.objectives, weights);
        let child_fit = self.tchebycheff_fitness(&child_sol.objectives, weights);
        if child_fit < cur_fit {
            child_sol.crowding_distance = child_fit;
            Ok(child_sol)
        } else {
            let mut cur_clone = current.clone();
            cur_clone.crowding_distance = cur_fit;
            Ok(cur_clone)
        }
    }

    fn select_neighbor(&self, index: usize) -> Result<usize> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        if self.population.len() <= 1 {
            return Ok(0);
        }
        let nbhood = 10.min(self.population.len());
        let start = index.saturating_sub(nbhood / 2);
        let end = (index + nbhood / 2).min(self.population.len() - 1);
        if end <= start {
            return Ok(if index > 0 { index - 1 } else { 0 });
        }
        let ni = rng_inst.random_range(start..=end);
        if ni == index && end > start {
            Ok(if ni == start { end } else { start })
        } else {
            Ok(ni)
        }
    }

    fn tchebycheff_fitness(&self, objectives: &[f64], weights: &[f64]) -> f64 {
        let mut max_diff = 0.0f64;
        for (i, (&v, &w)) in objectives.iter().zip(weights.iter()).enumerate() {
            let ideal = if self
                .config
                .objectives
                .get(i)
                .map(|o| o.minimize)
                .unwrap_or(true)
            {
                0.0
            } else {
                1.0
            };
            max_diff = max_diff.max(w * (v - ideal).abs());
        }
        max_diff
    }

    fn update_neighbors(&mut self, index: usize, solution: &MultiObjectiveSolution) -> Result<()> {
        let nbhood = 10.min(self.population.len());
        let start = index.saturating_sub(nbhood / 2);
        let end = (index + nbhood / 2).min(self.population.len());
        let wvecs = self.generate_weight_vectors()?;
        for i in start..end {
            if i != index && i < wvecs.len() {
                let w = wvecs[i].clone();
                let cur_fit = self.tchebycheff_fitness(&self.population[i].objectives, &w);
                let new_fit = self.tchebycheff_fitness(&solution.objectives, &w);
                if new_fit < cur_fit {
                    self.population[i] = solution.clone();
                }
            }
        }
        Ok(())
    }

    fn hypervolume_environmental_selection(
        &self,
        mut combined: Vec<MultiObjectiveSolution>,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        combined.sort_by(|a, b| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(self.config.population_size);
        Ok(combined)
    }

    fn evaluate_constraints(&self, solution: &MultiObjectiveSolution) -> Result<Vec<f64>> {
        let mut violations = Vec::new();
        for (i, obj) in self.config.objectives.iter().enumerate() {
            if let (Some(target), Some(tol)) = (obj.target, obj.tolerance) {
                let v = solution.objectives.get(i).copied().unwrap_or(0.0);
                violations.push(((v - target).abs() - tol).max(0.0));
            }
        }
        Ok(violations)
    }

    fn constraint_environmental_selection(
        &self,
        mut offspring: Vec<MultiObjectiveSolution>,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        let mut combined = self.population.clone();
        combined.append(&mut offspring);
        combined.sort_by(|a, b| {
            let av: f64 = a.constraint_violations.iter().sum();
            let bv: f64 = b.constraint_violations.iter().sum();
            if (av - bv).abs() > 1e-12 {
                av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.objectives
                    .first()
                    .copied()
                    .unwrap_or(0.0)
                    .partial_cmp(&b.objectives.first().copied().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        combined.truncate(self.config.population_size);
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_objective_config() {
        let config = MultiObjectiveConfig::default();
        assert_eq!(config.objectives.len(), 4);
        assert_eq!(config.population_size, 50);
    }

    #[test]
    fn test_solution_dominance() {
        let config = MultiObjectiveConfig::default();
        let arch1 = Arc::new(crate::nas::architecture_encoding::SequentialEncoding::new(
            vec![],
        ));
        let arch2 = Arc::new(crate::nas::architecture_encoding::SequentialEncoding::new(
            vec![],
        ));
        let sol1 = MultiObjectiveSolution::new(arch1, vec![0.9, 1000.0, 500.0, 5.0]);
        let sol2 = MultiObjectiveSolution::new(arch2, vec![0.8, 500.0, 250.0, 2.5]);
        assert!(!sol1.dominates(&sol2, &config));
        assert!(!sol2.dominates(&sol1, &config));
    }

    #[test]
    fn test_optimizer_creation() {
        let config = MultiObjectiveConfig::default();
        let optimizer = MultiObjectiveOptimizer::new(config);
        assert_eq!(optimizer.generation, 0);
        assert!(optimizer.pareto_front.is_empty());
    }
}
