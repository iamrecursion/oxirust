// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Merge pass of [`AdaptiveSph::adaptive_step`]: greedy grouping of nearby
//! same-level particles and application of all merges in one go.
//!
//! Candidates for a merge are found with one sparse spatial hash per
//! refinement level, so the pass costs `O(n k²)` for `k` neighbours per
//! particle instead of testing every particle against every other one.

use std::collections::BTreeMap;

use super::{AdaptiveSph, SphParticle};
use crate::neighbor::SpatialHash3D;

/// Largest number of particles merged into one (the 8 children of a split).
const MAX_MERGE_GROUP: usize = 8;

/// Spatial index of the particles of one refinement level.
struct LevelIndex {
    /// Indices into [`AdaptiveSph::particles`] of the particles of the level.
    members: Vec<usize>,
    /// Hash over the positions of `members` (local index `k` is
    /// `members[k]`); its cell size is the largest reach `2 h` of the level.
    hash: SpatialHash3D,
}

fn squared_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

impl AdaptiveSph {
    /// One spatial index per refinement level. A level whose particles all
    /// have a non-positive or non-finite smoothing length has no reach, hence
    /// no neighbours, and gets no index.
    fn level_indices(&self) -> BTreeMap<u32, LevelIndex> {
        let mut members_by_level: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (index, particle) in self.particles.iter().enumerate() {
            members_by_level
                .entry(particle.level)
                .or_default()
                .push(index);
        }
        members_by_level
            .into_iter()
            .filter_map(|(level, members)| {
                let reach = members
                    .iter()
                    .map(|&m| 2.0 * self.particles[m].smoothing_length)
                    .filter(|reach| reach.is_finite() && *reach > 0.0)
                    .fold(0.0_f64, f64::max);
                if reach <= 0.0 {
                    return None;
                }
                let positions: Vec<[f64; 3]> = members
                    .iter()
                    .map(|&m| self.particles[m].position)
                    .collect();
                let hash = SpatialHash3D::build(&positions, reach);
                Some((level, LevelIndex { members, hash }))
            })
            .collect()
    }

    /// The not yet merged particles of the same refinement level as particle
    /// `i` that are closer than `2 h_i` (the reach of its kernel), nearest
    /// first (ties by index). Particle `i` itself is not listed.
    fn merge_candidates(
        &self,
        levels: &BTreeMap<u32, LevelIndex>,
        i: usize,
        consumed: &[bool],
    ) -> Vec<usize> {
        let particle = &self.particles[i];
        let reach = 2.0 * particle.smoothing_length;
        if !(reach.is_finite() && reach > 0.0) {
            return Vec::new();
        }
        let Some(level) = levels.get(&particle.level) else {
            return Vec::new();
        };
        let mut found: Vec<(f64, usize)> = level
            .hash
            .find_neighbors(particle.position)
            .into_iter()
            .map(|local| level.members[local])
            .filter(|&j| j != i && !consumed[j])
            .map(|j| {
                (
                    squared_distance(particle.position, self.particles[j].position),
                    j,
                )
            })
            .filter(|&(distance_sq, _)| distance_sq < reach * reach)
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        found.into_iter().map(|(_, j)| j).collect()
    }

    /// Greedy grouping of the particles that want to merge.
    ///
    /// Seeds are visited in index order. A seed whose same-level neighbours
    /// within `2 h` (see [`Self::should_merge`]) allow a merge starts a group
    /// and takes up to seven of those neighbours, nearest first, provided each
    /// of them qualifies as well (judged with its own neighbour list). A group
    /// needs at least two members; its members are not available to later
    /// seeds. Every group is a set of indices into the *current* particle
    /// vector with the seed first, all members lying within `2 h` of the seed.
    pub(super) fn merge_groups(&self) -> Vec<Vec<usize>> {
        let levels = self.level_indices();
        let mut consumed = vec![false; self.particles.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for seed in 0..self.particles.len() {
            if consumed[seed] {
                continue;
            }
            let candidates = self.merge_candidates(&levels, seed, &consumed);
            if !self.should_merge(seed, &candidates) {
                continue;
            }
            let mut group = vec![seed];
            for &candidate in &candidates {
                if group.len() >= MAX_MERGE_GROUP {
                    break;
                }
                let own_neighbours = self.merge_candidates(&levels, candidate, &consumed);
                if self.should_merge(candidate, &own_neighbours) {
                    group.push(candidate);
                }
            }
            if group.len() > 1 {
                for &member in &group {
                    consumed[member] = true;
                }
                groups.push(group);
            }
        }
        groups
    }

    /// Merge every group of [`Self::merge_groups`] and return the number of
    /// groups merged.
    ///
    /// All merged particles are built from the original indices first; then
    /// the absorbed particles are dropped in one stable `retain` and the merged
    /// particles are appended in group order. No index is looked up after a
    /// removal, so groups with interleaved index ranges cannot go stale.
    pub(super) fn merge_pass(&mut self) -> usize {
        let groups = self.merge_groups();
        if groups.is_empty() {
            return 0;
        }
        let merged: Vec<SphParticle> = groups
            .iter()
            .map(|group| self.merged_particle(group))
            .collect();

        let mut absorbed = vec![false; self.particles.len()];
        for &member in groups.iter().flatten() {
            absorbed[member] = true;
        }
        let mut position = 0;
        self.particles.retain(|_| {
            let keep = !absorbed[position];
            position += 1;
            keep
        });
        self.particles.extend(merged);
        groups.len()
    }
}
