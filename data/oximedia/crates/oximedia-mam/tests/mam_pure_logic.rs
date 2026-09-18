//! Integration tests pinning the PURE / in-memory behaviour of the MAM
//! access-control, version-control, and workflow-trigger subsystems.
//!
//! Scope: only the parts that need NO PostgreSQL, NO Redis, and NO external
//! infrastructure are exercised here.
//!
//! * `permissions` — the DB-backed `PermissionManager` (RBAC over PostgreSQL)
//!   is DEFERRED (it requires a live database). The pure surfaces tested are
//!   the role→permission model (`SystemRole::default_permissions`, which is the
//!   only in-memory expression of role inheritance) and the `AbacEngine`, which
//!   is the documented precedence / deny-overrides mechanism that complements
//!   RBAC.
//! * `asset_versioning` — `VersionTree` is fully in-memory and is the real home
//!   of branch (`fork`) / merge / lineage / conflict-detection logic. (The
//!   sibling `version_control::VersionHistory` module only records a flat linear
//!   history and has no branch/merge concept, so the branch+merge requirements
//!   are pinned against `asset_versioning` where they actually live.)
//! * `workflow_trigger` — `TriggerRegistry` is fully in-memory; concurrency is
//!   exercised by sharing it behind an `Arc<Mutex<…>>` across OS threads.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread;

use oximedia_mam::asset_versioning::{
    AssetVersion, SemanticVersion, VersionTree, VersionTreeError,
};
use oximedia_mam::permissions::{
    AbacCondition, AbacContext, AbacEngine, AbacPolicy, AttributeSource, AttributeValue,
    ComparisonOp, Permission, PolicyEffect, SystemRole,
};
use oximedia_mam::workflow_trigger::{
    AssetEvent, TriggerAction, TriggerCondition, TriggerContext, TriggerRegistry, TriggerRule,
};
use uuid::Uuid;

// ===========================================================================
// 1. permissions RBAC — role hierarchy + inheritance
// ===========================================================================
//
// The in-memory expression of the role hierarchy is `SystemRole`: more
// privileged roles are supersets of less privileged ones. We pin that a more
// privileged role *inherits* (contains) every permission its less-privileged
// peers grant — i.e. Admin ⊇ Editor ⊇ Viewer ⊇ Guest for the permissions they
// share — which is the real "inheritance" guarantee of the role model.
#[test]
fn rbac_role_hierarchy_inherits_lower_role_permissions() {
    let admin: HashSet<Permission> = SystemRole::Admin
        .default_permissions()
        .into_iter()
        .collect();
    let editor: HashSet<Permission> = SystemRole::Editor
        .default_permissions()
        .into_iter()
        .collect();
    let viewer: HashSet<Permission> = SystemRole::Viewer
        .default_permissions()
        .into_iter()
        .collect();
    let guest: HashSet<Permission> = SystemRole::Guest
        .default_permissions()
        .into_iter()
        .collect();

    // Guest can only read assets; that read capability is inherited all the way
    // up the hierarchy.
    assert!(
        guest.contains(&Permission::AssetRead),
        "guest must have asset:read"
    );
    assert!(
        viewer.is_superset(&guest),
        "viewer must inherit every guest permission; viewer={viewer:?} guest={guest:?}"
    );
    assert!(
        editor.is_superset(&viewer),
        "editor must inherit every viewer permission; editor={editor:?} viewer={viewer:?}"
    );
    assert!(
        admin.is_superset(&editor),
        "admin must inherit every editor permission; admin={admin:?} editor={editor:?}"
    );

    // Transitivity: admin therefore inherits guest+viewer too.
    assert!(admin.is_superset(&viewer));
    assert!(admin.is_superset(&guest));

    // And the hierarchy is strict: each higher role grants something the lower
    // one does not (so inheritance is not a degenerate "all equal").
    assert!(
        editor.len() > viewer.len(),
        "editor should be strictly more capable than viewer"
    );
    assert!(
        admin.contains(&Permission::SystemAdmin) && !editor.contains(&Permission::SystemAdmin),
        "only admin should hold system:admin"
    );
    // A Custom role is a blank slate — no inherited defaults.
    assert!(
        SystemRole::Custom.default_permissions().is_empty(),
        "custom role must start with no permissions"
    );
}

// ===========================================================================
// 2. permissions precedence — deny-overrides via ABAC priority ordering
// ===========================================================================
//
// The documented precedence rule (permissions.rs `AbacEngine::evaluate`) is
// "lower priority number = evaluated first; first matching policy wins". We pin
// that a more-specific Deny placed at a *higher* precedence (lower number)
// overrides a broad inherited Allow for the same subject — i.e. an explicit
// deny beats an inherited grant.
#[test]
fn abac_explicit_deny_overrides_inherited_allow() {
    let mut engine = AbacEngine::new();

    // Broad, low-precedence grant: allow everyone (priority 200, evaluated last).
    engine.add_policy(AbacPolicy::new("allow_all", PolicyEffect::Allow).with_priority(200));

    // Specific, high-precedence deny: external/contractor users are denied
    // (priority 10, evaluated first).
    engine.add_policy(
        AbacPolicy::new("deny_contractors", PolicyEffect::Deny)
            .with_priority(10)
            .with_condition(AbacCondition {
                attribute_key: "employment".into(),
                attribute_source: AttributeSource::Subject,
                operator: ComparisonOp::Equals,
                expected: AttributeValue::Str("contractor".into()),
            }),
    );

    // A contractor: the specific Deny is evaluated before the broad Allow and
    // wins → access denied even though a blanket Allow exists.
    let contractor_ctx =
        AbacContext::new().with_subject("employment", AttributeValue::Str("contractor".into()));
    assert_eq!(
        engine.evaluate(&Permission::AssetDelete, &contractor_ctx),
        Some(PolicyEffect::Deny),
        "explicit higher-precedence deny must override the inherited blanket allow"
    );
    assert_eq!(
        engine.is_allowed(&Permission::AssetDelete, &contractor_ctx),
        Some(false)
    );

    // A full-time employee: the Deny condition does NOT match, so evaluation
    // falls through to the broad Allow → access granted. This proves the deny
    // is *targeted*, not global.
    let employee_ctx =
        AbacContext::new().with_subject("employment", AttributeValue::Str("fulltime".into()));
    assert_eq!(
        engine.evaluate(&Permission::AssetDelete, &employee_ctx),
        Some(PolicyEffect::Allow),
        "non-matching deny must let the inherited allow through"
    );
    assert_eq!(
        engine.is_allowed(&Permission::AssetDelete, &employee_ctx),
        Some(true)
    );

    // Sanity: insertion order must not matter — the engine sorts by priority,
    // so even though "allow_all" was inserted first it is evaluated last.
    let policies = engine.policies();
    assert_eq!(policies[0].name, "deny_contractors");
    assert_eq!(policies[1].name, "allow_all");
}

// ===========================================================================
// 3. version_control — branch + merge clean, lineage correct
// ===========================================================================
//
// Pinned against `asset_versioning::VersionTree`, the real branch/merge home:
// branch (fork) twice from a common base, diverge, merge back → assert the
// merge node records BOTH parents and the version lineage (ancestry path) is
// reconstructable from root to head.
#[test]
fn version_tree_branch_and_merge_clean_with_lineage() {
    let asset_id = Uuid::new_v4();
    let user = Uuid::new_v4();
    let mut tree = VersionTree::new(asset_id);

    // Base version (root of lineage).
    let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "v1.0.0 base", user);
    let root_id = root.id;
    tree.add_root(root).expect("add_root should succeed");

    // Diverge: two independent branches off the same base.
    let branch_a = tree
        .fork(root_id, "branch-a: colour grade", user)
        .expect("fork branch-a");
    let branch_b = tree
        .fork(root_id, "branch-b: audio mix", user)
        .expect("fork branch-b");

    // Both branches descend directly from root.
    assert_eq!(
        tree.get(&branch_a).expect("branch-a exists").parent_ids,
        vec![root_id]
    );
    assert_eq!(
        tree.get(&branch_b).expect("branch-b exists").parent_ids,
        vec![root_id]
    );
    // Forks default to a minor bump from the base 1.0.0.
    assert_eq!(
        tree.get(&branch_a).expect("branch-a exists").semver,
        SemanticVersion::new(1, 1, 0)
    );

    // Merge the two branches back into a single integration version.
    let merge_semver = SemanticVersion::new(2, 0, 0);
    let merge_id = tree
        .merge(
            branch_a,
            branch_b,
            merge_semver.clone(),
            "v2.0.0 integrated",
            user,
        )
        .expect("merge should succeed");

    // Merge node must record BOTH parents (no parent silently dropped).
    let merged = tree.get(&merge_id).expect("merge node exists");
    assert_eq!(
        merged.parent_ids.len(),
        2,
        "merge node must have exactly two parents"
    );
    assert!(merged.parent_ids.contains(&branch_a));
    assert!(merged.parent_ids.contains(&branch_b));
    assert_eq!(merged.semver, merge_semver);

    // Head advances to the merge node.
    assert_eq!(tree.head().map(|v| v.id), Some(merge_id));

    // Lineage: ancestry follows the first parent (left-biased) back to root.
    // branch_a was the first merge parent, so root → branch_a → merge.
    let lineage = tree.ancestry_path(merge_id);
    assert_eq!(
        lineage,
        vec![root_id, branch_a, merge_id],
        "lineage must run root → first-parent branch → merge"
    );

    // The whole tree holds exactly 4 nodes: root + 2 branches + 1 merge.
    assert_eq!(tree.len(), 4);
}

// ===========================================================================
// 4. version_control — merge conflict / divergence is not silently lost
// ===========================================================================
//
// `VersionTree::merge` rejects an attempt to merge parents that belong to
// DIFFERENT assets with `MergeAssetMismatch` — the conflict is surfaced as an
// error, not silently absorbed. Separately, when two same-asset branches
// diverge by editing the *same* metadata field to different values, the merge
// preserves BOTH divergent parents (each value remains reachable via its
// parent), so no edit is silently lost.
#[test]
fn version_tree_merge_conflict_detected_and_no_silent_loss() {
    let user = Uuid::new_v4();

    // --- Part A: cross-asset merge is rejected (conflict surfaced) ---
    let asset_x = Uuid::new_v4();
    let asset_y = Uuid::new_v4();
    let mut tree_x = VersionTree::new(asset_x);

    let root_x = AssetVersion::new_root(asset_x, SemanticVersion::initial(), "x-root", user);
    let root_x_id = root_x.id;
    tree_x.add_root(root_x).expect("add x root");
    let branch_x = tree_x.fork(root_x_id, "x-branch", user).expect("fork x");

    // A version belonging to a DIFFERENT asset is inserted as a stray root.
    let foreign = AssetVersion::new_root(asset_y, SemanticVersion::initial(), "y-root", user);
    let foreign_id = foreign.id;
    // Insert it directly into tree_x's node map via a second add_root so the
    // merge can look it up; its asset_id differs from the tree's asset.
    tree_x.add_root(foreign).expect("insert foreign node");

    let conflict = tree_x.merge(
        branch_x,
        foreign_id,
        SemanticVersion::new(2, 0, 0),
        "bad-merge",
        user,
    );
    assert!(
        matches!(conflict, Err(VersionTreeError::MergeAssetMismatch { .. })),
        "merging parents from different assets must be reported as a conflict, got {conflict:?}"
    );

    // --- Part B: same-asset divergent field edits both survive the merge ---
    let asset_id = Uuid::new_v4();
    let mut tree = VersionTree::new(asset_id);
    let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "base", user);
    let root_id = root.id;
    tree.add_root(root).expect("add base");

    // Two branches edit the SAME metadata key "grade" to DIFFERENT values.
    let branch_a = tree.fork(root_id, "grade=warm", user).expect("fork a");
    tree.get_mut(&branch_a)
        .expect("branch a")
        .metadata
        .insert("grade".into(), "warm".into());

    let branch_b = tree.fork(root_id, "grade=cool", user).expect("fork b");
    tree.get_mut(&branch_b)
        .expect("branch b")
        .metadata
        .insert("grade".into(), "cool".into());

    let merge_id = tree
        .merge(
            branch_a,
            branch_b,
            SemanticVersion::new(2, 0, 0),
            "merged grades",
            user,
        )
        .expect("same-asset merge should succeed");

    // Both divergent edits remain reachable through the merge node's parents —
    // neither value was silently overwritten or dropped.
    let merged = tree.get(&merge_id).expect("merge exists");
    let parent_a = tree
        .get(&merged.parent_ids[0])
        .expect("first parent exists");
    let parent_b = tree
        .get(&merged.parent_ids[1])
        .expect("second parent exists");
    let mut grades: HashSet<&str> = HashSet::new();
    if let Some(g) = parent_a.metadata.get("grade") {
        grades.insert(g.as_str());
    }
    if let Some(g) = parent_b.metadata.get("grade") {
        grades.insert(g.as_str());
    }
    assert_eq!(
        grades,
        HashSet::from(["warm", "cool"]),
        "both divergent field edits must survive in the merge lineage; got {grades:?}"
    );
}

// ===========================================================================
// 5. workflow_trigger — concurrent fire, no firing lost
// ===========================================================================
//
// `TriggerRegistry::fire` takes `&mut self`, so concurrent producers share it
// behind a Mutex. Fire N events from N threads; assert exactly N firings are
// recorded in the log (none lost, none duplicated) and the per-rule firing
// count equals N.
#[test]
fn trigger_registry_concurrent_fire_no_loss() {
    const N: usize = 64;

    let mut registry = TriggerRegistry::new();
    registry.register(TriggerRule::new(
        "ingest-rule",
        "fire on every ingest",
        AssetEvent::Ingested,
        TriggerCondition::Always,
        vec![TriggerAction::StartWorkflow("transcode".to_string())],
    ));
    let shared = Arc::new(Mutex::new(registry));

    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let reg = Arc::clone(&shared);
        handles.push(thread::spawn(move || {
            let ctx = TriggerContext::new(format!("asset-{i:03}"), "video", 1_000 + i as u64);
            let mut guard = reg.lock().expect("registry lock should not be poisoned");
            let firings = guard.fire(AssetEvent::Ingested, ctx);
            // Each fire of this single matching rule yields exactly one firing.
            assert_eq!(
                firings.len(),
                1,
                "each event must fire the rule exactly once"
            );
        }));
    }
    for h in handles {
        h.join().expect("worker thread should not panic");
    }

    let guard = shared.lock().expect("final lock");
    // Exactly N firings recorded — nothing lost to the race, nothing duplicated.
    assert_eq!(
        guard.firing_log().len(),
        N,
        "all {N} concurrent firings must be recorded"
    );
    assert_eq!(
        guard.firings_for_rule("ingest-rule").len(),
        N,
        "the rule must have fired exactly {N} times"
    );

    // Every distinct asset id appears exactly once (no firing collapsed or
    // double-counted).
    let distinct_assets: HashSet<&str> = guard
        .firings_for_rule("ingest-rule")
        .iter()
        .map(|f| f.context.asset_id.as_str())
        .collect();
    assert_eq!(
        distinct_assets.len(),
        N,
        "each of the {N} firings must carry a unique asset id"
    );
}

// ===========================================================================
// 6. workflow_trigger — repeat-fire semantics + ordering preserved
// ===========================================================================
//
// The real semantic of `TriggerRegistry::fire` is append-only: it is NOT
// idempotent — firing the SAME event twice records TWO firings (by design,
// every asset event is auditable). We pin that real behaviour: two firings of
// the same event produce two distinct log entries, and that the firing log
// preserves insertion order for the sequence of events.
#[test]
fn trigger_registry_repeat_fire_appends_and_preserves_order() {
    let mut registry = TriggerRegistry::new();
    registry.register(TriggerRule::new(
        "qc-rule",
        "audit on qc pass",
        AssetEvent::QcPassed,
        TriggerCondition::Always,
        vec![TriggerAction::AuditLog("qc passed".to_string())],
    ));

    // Fire the SAME logical event twice for the same asset.
    let ctx = TriggerContext::new("asset-dup", "video", 5_000);
    let first = registry.fire(AssetEvent::QcPassed, ctx.clone());
    let second = registry.fire(AssetEvent::QcPassed, ctx);
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);

    // Append-only: two firings recorded, not deduplicated to one.
    assert_eq!(
        registry.firings_for_rule("qc-rule").len(),
        2,
        "repeat-firing the same event must append (no silent dedup)"
    );

    // Ordering: a sequence of distinct events is recorded in fire() order.
    let mut ordered = TriggerRegistry::new();
    ordered.register(TriggerRule::new(
        "all-ingest",
        "ordered ingest audit",
        AssetEvent::Ingested,
        TriggerCondition::Always,
        vec![TriggerAction::AuditLog("ingest".to_string())],
    ));
    let sequence = ["alpha", "bravo", "charlie", "delta"];
    for (i, id) in sequence.iter().enumerate() {
        ordered.fire(
            AssetEvent::Ingested,
            TriggerContext::new(*id, "video", 1_000 + i as u64),
        );
    }
    let recorded: Vec<&str> = ordered
        .firing_log()
        .iter()
        .map(|f| f.context.asset_id.as_str())
        .collect();
    assert_eq!(
        recorded, sequence,
        "single-threaded firing log must preserve event order"
    );
}
