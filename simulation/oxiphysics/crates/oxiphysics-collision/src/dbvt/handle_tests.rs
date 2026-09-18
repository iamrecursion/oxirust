// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the generational leaf handles of [`DynamicBvh`].
//!
//! Leaves used to be addressed by raw node indices. The tree recycles the slots
//! of removed nodes, so an index that outlived its leaf silently named whatever
//! leaf was inserted into the slot next: removing it removed an unrelated proxy
//! without any diagnostic. A [`ProxyId`] pairs the slot with its generation,
//! which is incremented whenever the slot is freed.

use std::collections::BTreeMap;

use super::types::{BvhAabb, DynamicBvh, ProxyId};
use crate::Error;
use oxiphysics_core::Vec3;

fn cube(x: f64, y: f64, z: f64, half: f64) -> BvhAabb {
    BvhAabb::new(
        Vec3::new(x - half, y - half, z - half),
        Vec3::new(x + half, y + half, z + half),
    )
}

/// Small deterministic generator for the churn test.
struct Lcg(u64);

impl Lcg {
    fn next_bits(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f64 {
        self.next_bits() as f64 / (1u64 << 31) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_bits() % n as u64) as usize
    }
}

fn is_invalid_handle(result: &crate::Result<()>, index: usize) -> bool {
    matches!(result, Err(Error::InvalidHandle { index: i }) if *i == index)
}

#[test]
fn a_stale_handle_cannot_remove_the_proxy_that_reused_its_slot() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 10);
    bvh.remove(a);
    // The freed slot is recycled for the next leaf.
    let b = bvh.insert(cube(5.0, 0.0, 0.0, 1.0), 20);
    assert_eq!(b.index, a.index, "the slot of A is reused by B");
    assert_ne!(b.generation, a.generation, "but under a new generation");

    let outcome = bvh.try_remove(a);
    assert!(is_invalid_handle(&outcome, a.index as usize), "{outcome:?}");
    // B is untouched.
    assert_eq!(bvh.n_leaves(), 1);
    assert_eq!(bvh.query_aabb(&cube(5.0, 0.0, 0.0, 0.5)), vec![20]);
    assert!(bvh.leaf_aabb(b).is_some());
    assert!(bvh.validate());
    // B's own handle works.
    assert!(bvh.try_remove(b).is_ok());
    assert_eq!(bvh.n_leaves(), 0);
}

#[test]
fn a_stale_handle_cannot_move_the_proxy_that_reused_its_slot() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 10);
    bvh.remove(a);
    let b = bvh.insert(cube(5.0, 0.0, 0.0, 1.0), 20);
    let before = bvh.leaf_aabb(b).expect("B is live");

    let far = cube(100.0, 100.0, 100.0, 1.0);
    let outcome = bvh.try_update(a, far);
    assert!(
        matches!(outcome, Err(Error::InvalidHandle { index }) if index == a.index as usize),
        "{outcome:?}"
    );
    let moved = bvh.try_move_proxy(a, far, Vec3::new(1.0, 0.0, 0.0), 0.1);
    assert!(
        matches!(moved, Err(Error::InvalidHandle { .. })),
        "{moved:?}"
    );
    // B did not move.
    let after = bvh.leaf_aabb(b).expect("B is live");
    assert!(after.approx_eq(&before, 0.0), "{after:?} vs {before:?}");
    assert!(bvh.validate());
}

#[test]
fn removing_twice_is_an_error_and_leaves_the_tree_consistent() {
    let mut bvh = DynamicBvh::new();
    let ids: Vec<ProxyId> = (0..6)
        .map(|i| bvh.insert(cube(3.0 * f64::from(i), 0.0, 0.0, 1.0), i as u32))
        .collect();
    assert!(bvh.try_remove(ids[2]).is_ok());
    let again = bvh.try_remove(ids[2]);
    assert!(
        is_invalid_handle(&again, ids[2].index as usize),
        "{again:?}"
    );
    assert_eq!(bvh.n_leaves(), 5);
    assert!(bvh.validate());
    assert!(bvh.validate_heights());
}

#[test]
fn handles_that_were_never_issued_are_rejected() {
    let mut bvh = DynamicBvh::new();
    let live = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    // Far beyond the node storage: rejected, no out-of-range indexing.
    let wild = ProxyId {
        index: 1_000_000,
        generation: 0,
    };
    let outcome = bvh.try_remove(wild);
    assert!(is_invalid_handle(&outcome, 1_000_000), "{outcome:?}");
    assert!(bvh.try_update(wild, cube(1.0, 1.0, 1.0, 1.0)).is_err());
    assert!(bvh.leaf_aabb(wild).is_none());
    assert!(!bvh.relabel_leaf(wild, 5));
    // A right slot with the wrong generation is rejected as well.
    let forged = ProxyId {
        index: live.index,
        generation: live.generation.wrapping_add(7),
    };
    assert!(bvh.try_remove(forged).is_err());
    assert_eq!(bvh.n_leaves(), 1);
    assert!(bvh.try_remove(live).is_ok());
}

#[test]
fn the_slot_of_an_internal_node_is_not_a_leaf_handle() {
    let mut bvh = DynamicBvh::new();
    for i in 0..4 {
        bvh.insert(cube(3.0 * f64::from(i), 0.0, 0.0, 1.0), i as u32);
    }
    let root = bvh.root().expect("the tree has a root");
    assert!(
        !bvh.nodes[root].is_leaf(),
        "four leaves have an internal root"
    );
    // A forged handle that even matches the generation of the slot.
    let forged = ProxyId {
        index: root as u32,
        generation: bvh.nodes[root].generation,
    };
    let outcome = bvh.try_remove(forged);
    assert!(is_invalid_handle(&outcome, root), "{outcome:?}");
    assert!(bvh.leaf_aabb(forged).is_none());
    assert_eq!(bvh.n_leaves(), 4);
    assert!(bvh.validate());
}

#[test]
#[should_panic(expected = "DynamicBvh::remove")]
fn remove_panics_for_a_stale_handle_instead_of_corrupting_the_tree() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    bvh.remove(a);
    bvh.insert(cube(2.0, 0.0, 0.0, 1.0), 2);
    bvh.remove(a);
}

#[test]
#[should_panic(expected = "DynamicBvh::update")]
fn update_panics_for_a_stale_handle() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    bvh.remove(a);
    bvh.update(a, cube(9.0, 9.0, 9.0, 1.0));
}

#[test]
#[should_panic(expected = "DynamicBvh::move_proxy")]
fn move_proxy_panics_for_a_stale_handle() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    bvh.remove(a);
    bvh.move_proxy(a, cube(9.0, 9.0, 9.0, 1.0), Vec3::new(0.0, 0.0, 0.0), 0.0);
}

#[test]
#[should_panic(expected = "DynamicBvh::update_leaf_inplace")]
fn update_leaf_inplace_panics_for_a_stale_handle() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    bvh.remove(a);
    bvh.update_leaf_inplace(a, cube(0.0, 0.0, 0.0, 1.0));
}

#[test]
#[should_panic(expected = "DynamicBvh::refit_leaf_ancestors")]
fn refit_leaf_ancestors_panics_for_a_stale_handle() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    bvh.remove(a);
    bvh.refit_leaf_ancestors(a);
}

#[test]
fn clear_invalidates_every_handle_and_never_lets_one_alias_a_new_leaf() {
    let mut bvh = DynamicBvh::new();
    let old: Vec<ProxyId> = (0..5)
        .map(|i| bvh.insert(cube(2.0 * f64::from(i), 0.0, 0.0, 0.5), i as u32))
        .collect();
    bvh.clear();
    assert_eq!(bvh.n_leaves(), 0);
    assert_eq!(bvh.n_nodes(), 0);
    assert!(bvh.root().is_none());
    // New leaves reuse the old slots, under new generations.
    let fresh: Vec<ProxyId> = (0..5)
        .map(|i| {
            bvh.insert(
                cube(50.0 + 2.0 * f64::from(i), 0.0, 0.0, 0.5),
                100 + i as u32,
            )
        })
        .collect();
    for stale in &old {
        let outcome = bvh.try_remove(*stale);
        assert!(outcome.is_err(), "{stale:?} survived the clear");
        assert!(
            fresh.iter().all(|f| f != stale),
            "{stale:?} equals a new handle"
        );
    }
    assert_eq!(bvh.n_leaves(), 5);
    assert!(bvh.validate());
    for (i, id) in fresh.iter().enumerate() {
        assert!(bvh.leaf_aabb(*id).is_some(), "fresh handle {i}");
    }
}

#[test]
fn handles_survive_update_move_proxy_and_rebuild() {
    let mut rng = Lcg(99);
    let mut bvh = DynamicBvh::new();
    let ids: Vec<(ProxyId, u32)> = (0..40u32)
        .map(|data| {
            let center = (rng.unit() * 20.0, rng.unit() * 20.0, rng.unit() * 20.0);
            (
                bvh.insert(cube(center.0, center.1, center.2, 0.6), data),
                data,
            )
        })
        .collect();

    // A far move re-inserts the leaf: the handle keeps naming it.
    let (id, data) = ids[7];
    let target = cube(500.0, 500.0, 500.0, 0.6);
    assert!(bvh.update(id, target), "a far move must re-insert the leaf");
    assert_eq!(bvh.find_leaf(data), Some(id));
    assert!(bvh.leaf_aabb(id).expect("valid").contains(&target));
    assert!(bvh.validate());

    // A small move inside the fat box does nothing and keeps the handle.
    let nudged = bvh.try_update(id, target.translate(Vec3::new(0.01, 0.0, 0.0)));
    assert!(matches!(nudged, Ok(false)), "{nudged:?}");

    // Predicted motion re-inserts too.
    let moved = bvh
        .try_move_proxy(
            ids[3].0,
            cube(-300.0, 0.0, 0.0, 0.6),
            Vec3::new(-10.0, 0.0, 0.0),
            0.5,
        )
        .expect("valid handle");
    assert!(moved);
    assert_eq!(bvh.find_leaf(ids[3].1), Some(ids[3].0));

    // A full rebuild keeps every handle valid and the pair set unchanged.
    let pairs_before = bvh.query_overlapping_pairs();
    bvh.rebuild();
    assert!(bvh.validate());
    assert!(bvh.validate_heights());
    assert_eq!(bvh.n_leaves(), ids.len());
    assert_eq!(bvh.query_overlapping_pairs(), pairs_before);
    for &(id, data) in &ids {
        assert!(
            bvh.leaf_aabb(id).is_some(),
            "handle of {data} after rebuild"
        );
        assert_eq!(bvh.find_leaf(data), Some(id));
    }
    // ... and still removes the right leaf.
    assert!(bvh.try_remove(ids[11].0).is_ok());
    assert_eq!(bvh.find_leaf(ids[11].1), None);
    assert_eq!(bvh.n_leaves(), ids.len() - 1);
}

#[test]
fn updating_the_only_leaf_keeps_its_handle() {
    let mut bvh = DynamicBvh::new();
    let id = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 5);
    assert!(bvh.update(id, cube(40.0, 0.0, 0.0, 1.0)));
    assert_eq!(bvh.root(), Some(id.index as usize));
    assert_eq!(bvh.n_leaves(), 1);
    assert_eq!(bvh.query_aabb(&cube(40.0, 0.0, 0.0, 0.5)), vec![5]);
    assert!(bvh.try_remove(id).is_ok());
    assert!(bvh.root().is_none());
    assert!(bvh.try_remove(id).is_err());
}

#[test]
fn remove_where_and_insert_bulk_work_with_handles() {
    let mut bvh = DynamicBvh::new();
    let items: Vec<(BvhAabb, u32)> = (0..8u32)
        .map(|i| (cube(3.0 * f64::from(i), 0.0, 0.0, 1.0), i))
        .collect();
    let ids = bvh.insert_bulk(&items);
    assert_eq!(ids.len(), 8);
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), 8, "handles are distinct");

    assert_eq!(bvh.remove_where(|data| data % 2 == 1), 4);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(
            bvh.leaf_aabb(*id).is_some(),
            i % 2 == 0,
            "handle {i} after remove_where"
        );
    }
    assert!(bvh.validate());
}

#[test]
fn insert_and_try_insert_hand_out_the_same_kind_of_handle() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    let b = bvh
        .try_insert(cube(4.0, 0.0, 0.0, 1.0), 2)
        .expect("room for a leaf");
    assert_ne!(a, b);
    assert_eq!(bvh.find_leaf(1), Some(a));
    assert_eq!(bvh.find_leaf(2), Some(b));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn a_slot_index_beyond_the_handle_range_is_a_capacity_error() {
    let too_big = u32::MAX as usize + 1;
    let outcome = ProxyId::from_slot(too_big, 0);
    assert!(
        matches!(outcome, Err(Error::BroadphaseCapacity { capacity }) if capacity == u32::MAX as usize),
        "{outcome:?}"
    );
    let largest = ProxyId::from_slot(u32::MAX as usize, 3).expect("the largest index fits");
    assert_eq!(
        largest,
        ProxyId {
            index: u32::MAX,
            generation: 3
        }
    );
}

#[test]
fn serialize_exposes_the_slot_generation() {
    let mut bvh = DynamicBvh::new();
    let a = bvh.insert(cube(0.0, 0.0, 0.0, 1.0), 1);
    bvh.remove(a);
    let b = bvh.insert(cube(3.0, 0.0, 0.0, 1.0), 2);
    let buffer = bvh.serialize();
    let slot = b.index as usize;
    // 14 values per node after the leading count; the last one is the generation.
    let generation = buffer[1 + slot * 14 + 13];
    assert_eq!(generation, f64::from(b.generation));
    assert_eq!(generation, 1.0, "the slot was freed once");
}

#[test]
fn random_churn_never_confuses_live_and_stale_handles() {
    let mut rng = Lcg(2026);
    let mut bvh = DynamicBvh::new();
    let mut live: BTreeMap<ProxyId, u32> = BTreeMap::new();
    let mut stale: Vec<ProxyId> = Vec::new();
    let mut next_data = 0u32;
    let mut reused_slots = 0usize;

    for step in 0..4000 {
        match rng.below(10) {
            // Insert (more often than remove, so the tree grows).
            0..=4 => {
                let center = (rng.unit() * 30.0, rng.unit() * 30.0, rng.unit() * 30.0);
                let id = bvh.insert(
                    cube(center.0, center.1, center.2, 0.3 + rng.unit()),
                    next_data,
                );
                if stale.iter().any(|s| s.index == id.index) {
                    reused_slots += 1;
                }
                assert!(
                    live.insert(id, next_data).is_none(),
                    "duplicate handle {id:?}"
                );
                next_data += 1;
            }
            // Remove a live leaf through its handle.
            5..=7 if !live.is_empty() => {
                let id = *live.keys().nth(rng.below(live.len())).expect("non-empty");
                assert!(
                    bvh.try_remove(id).is_ok(),
                    "step {step}: live handle {id:?}"
                );
                live.remove(&id);
                stale.push(id);
            }
            // Move a live leaf.
            8..=9 if !live.is_empty() => {
                let id = *live.keys().nth(rng.below(live.len())).expect("non-empty");
                let center = (rng.unit() * 30.0, rng.unit() * 30.0, rng.unit() * 30.0);
                bvh.try_update(id, cube(center.0, center.1, center.2, 0.5))
                    .expect("live handle");
            }
            _ => {}
        }
        // Every so often a stale handle is presented: it must be refused and
        // must change nothing.
        if step % 7 == 0 && !stale.is_empty() {
            let id = stale[rng.below(stale.len())];
            let leaves = bvh.n_leaves();
            assert!(
                bvh.try_remove(id).is_err(),
                "step {step}: stale {id:?} accepted"
            );
            assert!(bvh.try_update(id, cube(1.0, 1.0, 1.0, 1.0)).is_err());
            assert_eq!(bvh.n_leaves(), leaves, "step {step}");
        }
    }
    assert!(
        reused_slots > 100,
        "only {reused_slots} slots were recycled"
    );
    assert_eq!(bvh.n_leaves(), live.len());
    assert!(bvh.validate());
    assert!(bvh.validate_heights());
    for (&id, &data) in &live {
        assert_eq!(bvh.find_leaf(data), Some(id), "payload {data}");
    }
    for id in &stale {
        assert!(
            !live.contains_key(id),
            "a stale handle {id:?} was handed out again"
        );
    }
}

#[test]
fn leaf_queries_count_only_live_slots_after_many_removals() {
    // 10 000 leaves on a 100 x 100 lattice, every second one removed: the free
    // list is long, and the leaf and node counts must still be exact.
    let mut bvh = DynamicBvh::new();
    let ids: Vec<ProxyId> = (0..10_000u32)
        .map(|i| {
            let x = f64::from(i % 100) * 3.0;
            let y = f64::from(i / 100) * 3.0;
            bvh.insert(cube(x, y, 0.0, 1.0), i)
        })
        .collect();
    for id in ids.iter().step_by(2) {
        assert!(bvh.try_remove(*id).is_ok());
    }
    assert_eq!(bvh.n_leaves(), 5_000);
    assert_eq!(bvh.n_internal(), 4_999);
    assert_eq!(bvh.n_nodes(), 9_999);
    assert_eq!(bvh.all_leaf_data().len(), 5_000);
    assert!(bvh.all_leaf_data().iter().all(|data| data % 2 == 1));
    assert_eq!(bvh.leaf_snapshot().len(), 5_000);
    assert!(bvh.validate() && bvh.validate_heights());

    // The nearest leaves to the origin are live ones, nearest first.
    let nearest = bvh.k_nearest(Vec3::new(0.0, 0.0, 0.0), 3);
    assert_eq!(nearest.len(), 3);
    assert_eq!(nearest[0].data, 1);
    assert!(nearest.iter().all(|leaf| leaf.data % 2 == 1));
    assert!(nearest.windows(2).all(|w| w[0].dist_sq <= w[1].dist_sq));

    // A freed slot has no depth; a live one has.
    assert_eq!(bvh.node_depth(ids[0].index as usize), None);
    assert!(bvh.node_depth(ids[1].index as usize).is_some());
    assert_eq!(bvh.node_depth(usize::MAX), None, "out of range");

    // Rotations only visit live internal nodes.
    bvh.optimize_rotations();
    bvh.balance_rotation();
    assert_eq!(bvh.n_leaves(), 5_000);
    assert!(bvh.validate());
}
