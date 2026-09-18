// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

// Regression tests of the merge pass of `AdaptiveSph::adaptive_step`. It used to
// merge up to eight same-level particles *anywhere* in the domain (no distance
// check), applied the merges one after the other with `swap_remove` + `push` so
// that later groups pointed at moved slots (out-of-bounds panic), and rebuilt an
// O(n) candidate list for every candidate (O(n³)). They live in their own file
// so that `adaptive.rs` stays below the 2000-line limit.

use super::*;

const BASE_H: f64 = 0.1;

fn child_h() -> f64 {
    AdaptiveSph::smoothing_length_for_level(BASE_H, 1)
}

fn particle(position: [f64; 3], velocity: [f64; 3], mass: f64, h: f64, level: u32) -> SphParticle {
    SphParticle::new(position, velocity, mass, 1000.0, 0.0, h, level)
}

fn total_mass(sph: &AdaptiveSph) -> f64 {
    sph.particles.iter().map(|p| p.mass).sum()
}

fn total_momentum(sph: &AdaptiveSph) -> [f64; 3] {
    let mut momentum = [0.0; 3];
    for p in &sph.particles {
        for (m, v) in momentum.iter_mut().zip(p.velocity.iter()) {
            *m += p.mass * v;
        }
    }
    momentum
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The eight corner offsets `±child_h/2` of a split.
fn corner_offsets() -> Vec<[f64; 3]> {
    let o = 0.5 * child_h();
    let mut offsets = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                offsets.push([sx * o, sy * o, sz * o]);
            }
        }
    }
    offsets
}

/// Two level-0 parents (masses 2 and 3) at the origin and at `separation` on
/// the x axis, both split into their eight level-1 children. The two splits
/// shuffle the particle vector (`swap_remove`), so the clusters' indices
/// interleave.
fn two_split_clusters(separation: f64) -> AdaptiveSph {
    let mut sph = AdaptiveSph::new(4, BASE_H);
    sph.add_particle(particle([0.0; 3], [0.0; 3], 2.0, BASE_H, 0));
    sph.add_particle(particle([separation, 0.0, 0.0], [0.0; 3], 3.0, BASE_H, 0));
    sph.split_particle(0);
    sph.split_particle(0);
    sph
}

#[test]
fn two_distant_clusters_merge_into_one_particle_each() {
    // Clusters 10 child smoothing lengths apart: far outside the 2h reach.
    let separation = 10.0 * child_h();
    let mut sph = two_split_clusters(separation);
    assert_eq!(sph.particles.len(), 16);
    let mass_before = total_mass(&sph);

    sph.adaptive_step();

    assert_eq!(sph.particles.len(), 2, "one merged particle per cluster");
    assert!((total_mass(&sph) - mass_before).abs() < 1.0e-12);
    let mut merged = sph.particles.clone();
    merged.sort_by(|a, b| a.position[0].total_cmp(&b.position[0]));
    // The children are placed symmetrically, so each centroid is the parent.
    assert!(distance(merged[0].position, [0.0; 3]) < 1.0e-12);
    assert!(distance(merged[1].position, [separation, 0.0, 0.0]) < 1.0e-12);
    assert!((merged[0].mass - 2.0).abs() < 1.0e-12);
    assert!((merged[1].mass - 3.0).abs() < 1.0e-12);
    for p in &merged {
        assert_eq!(p.level, 1);
        assert!((p.smoothing_length - child_h()).abs() < 1.0e-15);
    }
}

#[test]
fn interleaved_clusters_conserve_mass_and_merge_at_their_centres_of_mass() {
    // Cluster A (even indices) and cluster B (odd indices) with different
    // masses inside each cluster, so that the centre of mass differs from the
    // geometric centre; the groups' index ranges interleave completely.
    let offsets = corner_offsets();
    let centres = [[0.0, 0.0, 0.0], [10.0 * child_h(), 0.0, 0.0]];
    let mut sph = AdaptiveSph::new(4, BASE_H);
    let mut members: [Vec<(f64, [f64; 3])>; 2] = [Vec::new(), Vec::new()];
    for (k, offset) in offsets.iter().enumerate() {
        for (c, centre) in centres.iter().enumerate() {
            let position = [
                centre[0] + offset[0],
                centre[1] + offset[1],
                centre[2] + offset[2],
            ];
            let mass = 1.0 + k as f64 + 10.0 * c as f64;
            members[c].push((mass, position));
            sph.add_particle(particle(position, [0.0; 3], mass, child_h(), 1));
        }
    }
    let mass_before = total_mass(&sph);

    sph.adaptive_step();

    assert_eq!(sph.particles.len(), 2);
    assert!((total_mass(&sph) - mass_before).abs() < 1.0e-12);
    for (c, cluster) in members.iter().enumerate() {
        let mass: f64 = cluster.iter().map(|(m, _)| m).sum();
        let mut com = [0.0; 3];
        for (m, p) in cluster {
            for d in 0..3 {
                com[d] += m * p[d] / mass;
            }
        }
        let merged = sph
            .particles
            .iter()
            .find(|p| distance(p.position, centres[c]) < child_h())
            .expect("a merged particle in every cluster");
        assert!((merged.mass - mass).abs() < 1.0e-12, "cluster {c} mass");
        assert!(
            distance(merged.position, com) < 1.0e-12,
            "cluster {c}: merged at {:?}, centre of mass {com:?}",
            merged.position
        );
        // ... and inside the bounding box of its own cluster.
        for d in 0..3 {
            let lo = cluster
                .iter()
                .map(|(_, p)| p[d])
                .fold(f64::INFINITY, f64::min);
            let hi = cluster
                .iter()
                .map(|(_, p)| p[d])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                merged.position[d] >= lo && merged.position[d] <= hi,
                "cluster {c}, axis {d}: {} outside [{lo}, {hi}]",
                merged.position[d]
            );
        }
    }
}

#[test]
fn alternating_levels_do_not_panic_and_conserve_mass() {
    // Levels [0, 1, 0, 1, 0, 1] on a line. The former pass grouped all level-0
    // particles {0, 2, 4} and all level-1 particles {1, 3, 5}, merged {1, 3, 5}
    // first (swap_remove) and then indexed slot 4 of a 4-element vector.
    let mut sph = AdaptiveSph::new(4, BASE_H);
    for k in 0..6_usize {
        sph.add_particle(particle(
            [0.05 * k as f64, 0.0, 0.0],
            [0.0; 3],
            1.0 + k as f64,
            0.06,
            (k % 2) as u32,
        ));
    }
    let mass_before = total_mass(&sph);

    sph.adaptive_step();

    assert!((total_mass(&sph) - mass_before).abs() < 1.0e-12);
    // Same-level particles are 0.1 apart (< 2h = 0.12) only for the pairs
    // {0, 2} and {1, 3}; particles 4 and 5 have no free partner left.
    assert_eq!(sph.particles.len(), 4);
    let remaining: Vec<[f64; 3]> = sph.particles[..2].iter().map(|p| p.position).collect();
    assert_eq!(remaining, vec![[0.2, 0.0, 0.0], [0.25, 0.0, 0.0]]);
    // Merged particles are appended in group order.
    let m02 = &sph.particles[2];
    let m13 = &sph.particles[3];
    assert_eq!((m02.level, m13.level), (0, 1));
    assert!((m02.mass - 4.0).abs() < 1.0e-12);
    assert!((m13.mass - 6.0).abs() < 1.0e-12);
    assert!((m02.position[0] - (3.0 * 0.10) / 4.0).abs() < 1.0e-12);
    assert!((m13.position[0] - (2.0 * 0.05 + 4.0 * 0.15) / 6.0).abs() < 1.0e-12);
}

#[test]
fn distant_same_level_particles_are_not_merged() {
    let mut sph = AdaptiveSph::new(4, BASE_H);
    sph.add_particle(particle([0.0; 3], [0.0; 3], 1.0, BASE_H, 0));
    sph.add_particle(particle(
        [10.0 * BASE_H, 0.0, 0.0],
        [0.0; 3],
        1.0,
        BASE_H,
        0,
    ));
    let before = sph.particles.clone();

    sph.adaptive_step();

    assert_eq!(sph.particles.len(), 2);
    for (after, before) in sph.particles.iter().zip(before.iter()) {
        assert_eq!(after.position, before.position);
        assert_eq!(after.mass, before.mass);
    }
}

#[test]
fn isolated_particle_next_to_a_cluster_is_left_alone() {
    let mut sph = AdaptiveSph::new(4, BASE_H);
    let isolated = particle([5.0, 5.0, 5.0], [0.0; 3], 0.7, child_h(), 1);
    sph.add_particle(isolated.clone());
    sph.add_particle(particle([0.0; 3], [0.0; 3], 2.0, BASE_H, 0));
    sph.split_particle(1);
    assert_eq!(sph.particles.len(), 9);

    sph.adaptive_step();

    assert_eq!(sph.particles.len(), 2);
    // The isolated particle is kept (survivors keep their order, merged
    // particles are appended), the cluster collapsed into one particle.
    assert_eq!(sph.particles[0].position, isolated.position);
    assert_eq!(sph.particles[0].mass, isolated.mass);
    assert!((sph.particles[1].mass - 2.0).abs() < 1.0e-12);
    assert!(distance(sph.particles[1].position, [0.0; 3]) < 1.0e-12);
}

#[test]
fn a_group_never_exceeds_eight_particles() {
    // Twelve particles in a tight 2×2×3 block, all within 2h of each other.
    let mut sph = AdaptiveSph::new(4, BASE_H);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                sph.add_particle(particle(
                    [0.02 * i as f64, 0.02 * j as f64, 0.02 * k as f64],
                    [0.0; 3],
                    1.0,
                    BASE_H,
                    0,
                ));
            }
        }
    }
    let groups = sph.merge_groups();
    let mut sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![4, 8]);
    // The groups partition the particles.
    let mut all: Vec<usize> = groups.iter().flatten().copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..12).collect::<Vec<_>>());

    sph.adaptive_step();
    assert_eq!(sph.particles.len(), 2);
    let mut masses: Vec<f64> = sph.particles.iter().map(|p| p.mass).collect();
    masses.sort_by(f64::total_cmp);
    assert_eq!(masses, vec![4.0, 8.0]);
}

#[test]
fn every_group_member_lies_within_the_reach_of_its_seed() {
    // A line of same-level particles spaced 0.7 h: only neighbours within 2h of
    // the seed (the nearest two on each side) may join a group.
    let h = 0.05;
    let mut sph = AdaptiveSph::new(4, BASE_H);
    for k in 0..40_usize {
        sph.add_particle(particle(
            [0.7 * h * k as f64, 0.0, 0.0],
            [0.0; 3],
            1.0,
            h,
            0,
        ));
    }
    let groups = sph.merge_groups();
    assert!(!groups.is_empty());
    for group in &groups {
        let seed = sph.particles[group[0]].position;
        assert!(group.len() <= 8);
        for &member in &group[1..] {
            let d = distance(seed, sph.particles[member].position);
            assert!(
                d < 2.0 * h,
                "member {member} is {d} from its seed, reach {}",
                2.0 * h
            );
        }
    }
    let mass_before = total_mass(&sph);
    sph.adaptive_step();
    assert!((total_mass(&sph) - mass_before).abs() < 1.0e-12);
    assert!(sph.particles.len() < 40);
}

#[test]
fn momentum_is_conserved_and_each_cluster_keeps_its_velocity() {
    let mut sph = two_split_clusters(10.0 * child_h());
    // Uniform velocity per cluster: no velocity gradient inside a cluster.
    for p in &mut sph.particles {
        p.velocity = if p.position[0] < 0.5 * 10.0 * child_h() {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 2.0, 0.0]
        };
    }
    let momentum_before = total_momentum(&sph);
    let mass_before = total_mass(&sph);

    sph.adaptive_step();

    assert_eq!(sph.particles.len(), 2);
    assert!((total_mass(&sph) - mass_before).abs() < 1.0e-12);
    let momentum_after = total_momentum(&sph);
    for d in 0..3 {
        assert!((momentum_after[d] - momentum_before[d]).abs() < 1.0e-12);
    }
    for p in &sph.particles {
        let expected = if p.position[0] < 0.5 * 10.0 * child_h() {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 2.0, 0.0]
        };
        assert!(distance(p.velocity, expected) < 1.0e-12);
    }
}

#[test]
fn a_cluster_with_a_large_velocity_gradient_is_not_merged() {
    // Cluster A is at rest (merges); cluster B contracts quickly (gradient far
    // above merge_threshold) and, at max_level, is not split either.
    let mut sph = two_split_clusters(10.0 * child_h());
    sph.max_level = 1;
    for p in &mut sph.particles {
        if p.position[0] > 0.5 * 10.0 * child_h() {
            let centre = [10.0 * child_h(), 0.0, 0.0];
            let offset = [
                p.position[0] - centre[0],
                p.position[1] - centre[1],
                p.position[2] - centre[2],
            ];
            p.velocity = [-50.0 * offset[0], -50.0 * offset[1], -50.0 * offset[2]];
        }
    }

    sph.adaptive_step();

    let near_a = sph
        .particles
        .iter()
        .filter(|p| p.position[0] < 0.5 * 10.0 * child_h())
        .count();
    let near_b = sph.particles.len() - near_a;
    assert_eq!(near_a, 1, "the quiet cluster collapses to one particle");
    assert_eq!(near_b, 8, "the strongly sheared cluster stays resolved");
}

#[test]
fn merge_pass_is_deterministic() {
    let mut first = two_split_clusters(10.0 * child_h());
    let mut second = two_split_clusters(10.0 * child_h());
    first.adaptive_step();
    second.adaptive_step();
    assert_eq!(first.particles.len(), second.particles.len());
    for (a, b) in first.particles.iter().zip(second.particles.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.level, b.level);
    }
}

#[test]
fn many_clusters_merge_in_linear_work() {
    // 6×6×6 = 216 well separated clusters of eight children (1728 particles).
    // The former candidate search rebuilt an O(n) list for every candidate of
    // every seed (≈ n³ operations); the hashed search touches each particle's
    // neighbourhood only.
    let spacing = 10.0 * child_h();
    let mut sph = AdaptiveSph::new(4, BASE_H);
    let mut parents = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                let position = [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing];
                parents.push(position);
                sph.add_particle(particle(position, [0.0; 3], 1.0, BASE_H, 0));
            }
        }
    }
    // Split from the back so the parents' indices stay valid until used.
    for index in (0..parents.len()).rev() {
        sph.split_particle(index);
    }
    assert_eq!(sph.particles.len(), 216 * 8);

    let groups = sph.merge_groups();
    assert_eq!(groups.len(), 216);
    assert!(groups.iter().all(|g| g.len() == 8));

    sph.adaptive_step();
    assert_eq!(sph.particles.len(), 216);
    assert!((total_mass(&sph) - 216.0).abs() < 1.0e-9);
    for parent in &parents {
        assert!(
            sph.particles
                .iter()
                .any(|p| distance(p.position, *parent) < 1.0e-12 && (p.mass - 1.0).abs() < 1.0e-12),
            "no merged particle at {parent:?}"
        );
    }
}

#[test]
fn degenerate_smoothing_lengths_and_positions_do_not_panic_or_merge() {
    let mut sph = AdaptiveSph::new(4, BASE_H);
    sph.add_particle(particle([0.0; 3], [0.0; 3], 1.0, 0.0, 0));
    sph.add_particle(particle([0.01, 0.0, 0.0], [0.0; 3], 1.0, 0.0, 0));
    sph.add_particle(particle([0.02, 0.0, 0.0], [0.0; 3], 1.0, f64::NAN, 0));
    sph.add_particle(particle([0.03, 0.0, 0.0], [0.0; 3], 1.0, -0.5, 0));
    sph.add_particle(particle([f64::INFINITY, 0.0, 0.0], [0.0; 3], 1.0, 0.1, 0));
    sph.add_particle(particle([f64::NAN, 0.0, 0.0], [0.0; 3], 1.0, 0.1, 0));
    assert!(sph.merge_groups().is_empty());
    sph.adaptive_step();
    assert_eq!(sph.particles.len(), 6);
}

#[test]
fn merge_particles_counts_a_duplicated_index_once() {
    let mut sph = AdaptiveSph::new(4, BASE_H);
    sph.add_particle(particle([0.0; 3], [0.0; 3], 1.0, BASE_H, 0));
    sph.add_particle(particle([0.2, 0.0, 0.0], [0.0; 3], 3.0, BASE_H, 0));
    sph.add_particle(particle([9.0, 0.0, 0.0], [0.0; 3], 5.0, BASE_H, 0));

    let merged = sph.merge_particles(&[1, 0, 1]);

    assert_eq!(sph.particles.len(), 2);
    assert_eq!(merged, 1);
    let m = &sph.particles[merged];
    assert!((m.mass - 4.0).abs() < 1.0e-15);
    assert!((m.position[0] - 0.15).abs() < 1.0e-15);
    // The untouched far particle is still there.
    assert!(
        sph.particles
            .iter()
            .any(|p| p.position[0] == 9.0 && p.mass == 5.0)
    );
}

#[test]
fn merging_massless_particles_gives_a_finite_centroid() {
    let mut sph = AdaptiveSph::new(4, BASE_H);
    sph.add_particle(particle([0.0; 3], [1.0, 0.0, 0.0], 0.0, BASE_H, 0));
    sph.add_particle(particle([0.2, 0.0, 0.0], [3.0, 0.0, 0.0], 0.0, BASE_H, 0));
    let merged = sph.merge_particles(&[0, 1]);
    let m = &sph.particles[merged];
    assert_eq!(m.mass, 0.0);
    assert!((m.position[0] - 0.1).abs() < 1.0e-15);
    assert!((m.velocity[0] - 2.0).abs() < 1.0e-15);
    assert!(
        m.position
            .iter()
            .chain(m.velocity.iter())
            .all(|c| c.is_finite())
    );
}
