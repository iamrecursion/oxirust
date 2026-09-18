//! Regression tests for group removal and `CheckpointManager` eviction: an
//! evicted checkpoint must really disappear from the file, not just from the
//! manager's bookkeeping.

use crate::hdf5_io::*;

/// A checkpoint of `n_atoms` atoms whose values encode `step`.
fn sample_checkpoint(step: u64, n_atoms: usize) -> Checkpoint {
    let mut ckpt = Checkpoint::new(&format!("step_{step}"), step, step as f64 * 0.5);
    ckpt.positions = vec![step as f64; n_atoms * 3];
    ckpt.velocities = vec![-(step as f64); n_atoms * 3];
    ckpt.box_vectors = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    ckpt
}

/// Number of `f64` values stored in the datasets of `group` and below.
fn stored_f64_count(group: &Hdf5Group) -> usize {
    let own: usize = group
        .datasets
        .values()
        .map(|ds| match &ds.data {
            DataStorage::Float64(values) => values.len(),
            _ => 0,
        })
        .sum();
    own + group.groups.values().map(stored_f64_count).sum::<usize>()
}

/// `f64` values one checkpoint of `n_atoms` atoms occupies: positions,
/// velocities and the 3x3 box.
fn values_per_checkpoint(n_atoms: usize) -> usize {
    2 * n_atoms * 3 + 9
}

#[test]
fn checkpoint_manager_eviction_removes_group_from_file() {
    let n_atoms = 1000;
    let mut mgr = CheckpointManager::new(2);
    let mut file = Hdf5File::new("ckpt.h5");
    for step in 0..5_u64 {
        mgr.write(&mut file, &sample_checkpoint(step, n_atoms))
            .unwrap();
    }

    // The bookkeeping and the file agree: only the two newest checkpoints remain.
    let checkpoints = file.open_group("checkpoints").unwrap();
    assert_eq!(checkpoints.n_groups(), 2);
    assert_eq!(
        checkpoints.list_contents(),
        vec!["step_3".to_string(), "step_4".to_string()]
    );
    assert_eq!(mgr.count(), 2);
    let labels: Vec<&str> = mgr.checkpoint_labels.iter().map(String::as_str).collect();
    assert_eq!(labels, vec!["step_3", "step_4"]);

    // The evicted data is gone, so the memory is really released.
    for evicted in 0..3 {
        assert!(matches!(
            file.open_group(&format!("checkpoints/step_{evicted}")),
            Err(Hdf5Error::NotFound(_))
        ));
        assert!(matches!(
            Checkpoint::read_from_file(&file, &format!("step_{evicted}")),
            Err(Hdf5Error::NotFound(_))
        ));
    }
    assert_eq!(
        stored_f64_count(&file.root),
        2 * values_per_checkpoint(n_atoms)
    );

    // The survivors are intact.
    let latest = mgr.read_latest(&file).unwrap();
    assert_eq!(latest.label, "step_4");
    assert_eq!(latest.step, 4);
    assert_eq!(latest.positions, vec![4.0; n_atoms * 3]);
    assert_eq!(latest.velocities, vec![-4.0; n_atoms * 3]);
    let previous = Checkpoint::read_from_file(&file, "step_3").unwrap();
    assert_eq!(previous.positions, vec![3.0; n_atoms * 3]);
}

#[test]
fn checkpoint_manager_file_size_stays_bounded() {
    let n_atoms = 64;
    let max = 3;
    let mut mgr = CheckpointManager::new(max);
    let mut file = Hdf5File::new("long_run.h5");
    let mut peak = 0usize;
    for step in 0..60_u64 {
        mgr.write(&mut file, &sample_checkpoint(step, n_atoms))
            .unwrap();
        peak = peak.max(stored_f64_count(&file.root));
        assert!(mgr.count() <= max);
    }
    assert_eq!(peak, max * values_per_checkpoint(n_atoms));
    assert_eq!(
        file.open_group("checkpoints").unwrap().n_groups(),
        max,
        "one group per retained checkpoint"
    );
}

#[test]
fn checkpoint_manager_zero_capacity_keeps_nothing() {
    let mut mgr = CheckpointManager::new(0);
    let mut file = Hdf5File::new("none.h5");
    for step in 0..3_u64 {
        mgr.write(&mut file, &sample_checkpoint(step, 4)).unwrap();
    }
    assert_eq!(mgr.count(), 0);
    assert_eq!(file.open_group("checkpoints").unwrap().n_groups(), 0);
    assert_eq!(stored_f64_count(&file.root), 0);
    assert!(mgr.read_latest(&file).is_err());
}

#[test]
fn checkpoint_manager_shrinking_capacity_evicts_several() {
    let mut mgr = CheckpointManager::new(5);
    let mut file = Hdf5File::new("shrink.h5");
    for step in 0..5_u64 {
        mgr.write(&mut file, &sample_checkpoint(step, 2)).unwrap();
    }
    assert_eq!(mgr.count(), 5);

    // Lowering the cap is honoured by the next write, which evicts every
    // checkpoint above it, not just one.
    mgr.max_checkpoints = 2;
    mgr.write(&mut file, &sample_checkpoint(5, 2)).unwrap();
    assert_eq!(mgr.count(), 2);
    assert_eq!(
        file.open_group("checkpoints").unwrap().list_contents(),
        vec!["step_4".to_string(), "step_5".to_string()]
    );
}

#[test]
fn checkpoint_manager_tolerates_externally_removed_group() {
    let mut mgr = CheckpointManager::new(2);
    let mut file = Hdf5File::new("external.h5");
    for step in 0..2_u64 {
        mgr.write(&mut file, &sample_checkpoint(step, 2)).unwrap();
    }
    // Someone else deleted the oldest checkpoint already.
    file.remove_group("checkpoints/step_0").unwrap();

    mgr.write(&mut file, &sample_checkpoint(2, 2)).unwrap();
    let labels: Vec<&str> = mgr.checkpoint_labels.iter().map(String::as_str).collect();
    assert_eq!(labels, vec!["step_1", "step_2"]);
    assert_eq!(file.open_group("checkpoints").unwrap().n_groups(), 2);
}

#[test]
fn checkpoint_manager_rejects_labels_without_a_path_component() {
    let mut mgr = CheckpointManager::new(2);
    let mut file = Hdf5File::new("labels.h5");
    mgr.write(&mut file, &sample_checkpoint(1, 2)).unwrap();

    for label in ["", "/", "//"] {
        let mut ckpt = sample_checkpoint(9, 2);
        ckpt.label = label.to_string();
        assert!(
            matches!(mgr.write(&mut file, &ckpt), Err(Hdf5Error::Generic(_))),
            "label {label:?} must be rejected"
        );
    }
    // The valid checkpoint is untouched and nothing was recorded for the bad ones.
    assert_eq!(mgr.count(), 1);
    assert_eq!(Checkpoint::read_from_file(&file, "step_1").unwrap().step, 1);
}

#[test]
fn checkpoint_manager_failed_write_leaves_no_partial_group() {
    let mut mgr = CheckpointManager::new(4);
    let mut file = Hdf5File::new("partial.h5");
    mgr.write(&mut file, &sample_checkpoint(0, 2)).unwrap();

    // Two atoms of positions but only one atom of velocities: the positions
    // dataset is created before the velocities write is rejected.
    let mut broken = sample_checkpoint(1, 2);
    broken.velocities = vec![0.0; 3];
    let err = mgr.write(&mut file, &broken).unwrap_err();
    assert!(matches!(err, Hdf5Error::ShapeMismatch { .. }), "{err:?}");

    assert!(matches!(
        file.open_group("checkpoints/step_1"),
        Err(Hdf5Error::NotFound(_))
    ));
    assert_eq!(mgr.count(), 1);
    assert_eq!(stored_f64_count(&file.root), values_per_checkpoint(2));
}

#[test]
fn checkpoint_manager_duplicate_label_does_not_destroy_the_original() {
    let mut mgr = CheckpointManager::new(4);
    let mut file = Hdf5File::new("dup.h5");
    mgr.write(&mut file, &sample_checkpoint(7, 3)).unwrap();

    let err = mgr.write(&mut file, &sample_checkpoint(7, 3)).unwrap_err();
    assert!(matches!(err, Hdf5Error::AlreadyExists(_)), "{err:?}");

    assert_eq!(mgr.count(), 1);
    let restored = Checkpoint::read_from_file(&file, "step_7").unwrap();
    assert_eq!(restored.positions, vec![7.0; 9]);
    assert_eq!(restored.velocities, vec![-7.0; 9]);
}

#[test]
fn checkpoint_manager_locked_file_is_not_modified() {
    let mut mgr = CheckpointManager::new(1);
    let mut file = Hdf5File::new("locked.h5");
    mgr.write(&mut file, &sample_checkpoint(0, 2)).unwrap();

    file.lock_write(42).unwrap();
    let err = mgr.write(&mut file, &sample_checkpoint(1, 2)).unwrap_err();
    assert_eq!(err, Hdf5Error::FileLocked);
    file.unlock();

    assert_eq!(mgr.count(), 1);
    assert_eq!(
        file.open_group("checkpoints").unwrap().list_contents(),
        vec!["step_0".to_string()]
    );
}

#[test]
fn file_remove_group_removes_subtree_and_returns_it() {
    let mut file = Hdf5File::new("tree.h5");
    file.create_group("a/b/c").unwrap();
    file.create_dataset("a/b/c", "data", vec![4], Hdf5Dtype::Float64)
        .unwrap();
    file.create_group("a/keep").unwrap();

    let removed = file.remove_group("a/b").unwrap();
    assert_eq!(removed.name, "b");
    let removed_c = removed.open_group("c").unwrap();
    assert_eq!(removed_c.open_dataset("data").unwrap().volume(), 4);

    assert!(matches!(
        file.open_group("a/b"),
        Err(Hdf5Error::NotFound(_))
    ));
    assert!(matches!(
        file.open_group("a/b/c"),
        Err(Hdf5Error::NotFound(_))
    ));
    // Siblings and ancestors survive.
    assert!(file.open_group("a/keep").is_ok());
    assert_eq!(file.open_group("a").unwrap().n_groups(), 1);
    assert_eq!(stored_f64_count(&file.root), 0);
}

#[test]
fn file_remove_group_accepts_slashes_like_other_path_apis() {
    let mut file = Hdf5File::new("slashes.h5");
    file.create_group("x/y").unwrap();
    assert!(file.remove_group("/x//y/").is_ok());
    assert!(file.open_group("x").is_ok());
    assert!(file.remove_group("x").is_ok());
    assert_eq!(file.root.n_groups(), 0);
}

#[test]
fn file_remove_group_errors() {
    let mut file = Hdf5File::new("errors.h5");
    file.create_group("a/b").unwrap();

    // Missing leaf, missing intermediate, and a second removal.
    assert!(matches!(
        file.remove_group("a/missing"),
        Err(Hdf5Error::NotFound(_))
    ));
    assert!(matches!(
        file.remove_group("nope/b"),
        Err(Hdf5Error::NotFound(_))
    ));
    file.remove_group("a/b").unwrap();
    assert!(matches!(
        file.remove_group("a/b"),
        Err(Hdf5Error::NotFound(_))
    ));

    // The root group cannot be removed.
    for root in ["", "/", "//"] {
        assert!(matches!(
            file.remove_group(root),
            Err(Hdf5Error::Generic(_))
        ));
    }

    // A write lock blocks removal and leaves the group in place.
    file.lock_write(1).unwrap();
    assert_eq!(file.remove_group("a").unwrap_err(), Hdf5Error::FileLocked);
    file.unlock();
    assert!(file.open_group("a").is_ok());
}

#[test]
fn group_remove_group_only_detaches_child_groups() {
    let mut group = Hdf5Group::new("g");
    group.create_group("child").unwrap();
    group
        .create_dataset("child", vec![2], Hdf5Dtype::Float64)
        .unwrap();
    group.set_attr("note", AttrValue::Int32(3));

    let child = group.remove_group("child").unwrap();
    assert_eq!(child.name, "child");
    assert_eq!(group.n_groups(), 0);
    // The dataset that shares the name belongs to `group` itself and stays.
    assert_eq!(group.n_datasets(), 1);
    assert!(group.get_attr("note").is_ok());
    assert!(matches!(
        group.remove_group("child"),
        Err(Hdf5Error::NotFound(_))
    ));
}
