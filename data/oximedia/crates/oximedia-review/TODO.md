# oximedia-review TODO

## Current Status
- 46 modules (12 subdirectory modules) covering review sessions, annotations, approval workflows, comments/threads, version comparison, drawing tools, export (PDF/CSV/EDL), notifications, real-time collaboration, task tracking
- Core types: SessionId, CommentId, DrawingId, TaskId, VersionId, SessionConfig/Builder, User/UserRole, AnnotationType, WorkflowType
- Dependencies: oximedia-core, oximedia-timecode, serde, chrono, uuid, tokio

## Enhancements
- [ ] Add `SessionConfigBuilder::build()` validation that returns Result instead of using `unwrap_or_default` for required fields (verified-open 2026-05-16: not yet changed to return Result)
- [x] Extend `approval_workflow` with conditional approval rules (e.g., auto-approve if all issues marked resolved) (verified 2026-05-16; src/approval_workflow.rs:141 auto_approve_on_timeout, conditional rules:265)
- [x] Add `comment_thread` resolution tracking (open/resolved/wont-fix) with audit trail (verified 2026-05-16; src/comment_thread.rs:1 CommentStatus open/resolved/wont-fix, ThreadResolutionState:24)
- [x] Implement `review_metrics` with cycle time analytics (time from submission to approval per stage) (verified 2026-05-16; src/review_metrics.rs:343 CycleTimeStats, 712 lines)
- [x] Extend `drawing` tools with measurement annotations (rulers, angle markers, safe-area overlays) (verified 2026-05-16; src/drawing.rs:14 Ruler, AngleMarker, SafeAreaOverlay re-exports)
- [x] Add `review_diff` visual diff highlighting for comparing version changes (pixel-level diff overlay) (verified 2026-05-16; src/review_diff.rs:563 lines ReviewDiff, src/compare.rs:113 pixel-diff stats)
- [x] Implement `feedback_round` automatic numbering and deadline tracking per round (verified 2026-05-16; src/feedback_round.rs:176 FeedbackRound, FeedbackRoundManager:257)
- [x] Add `review_tag` hierarchical tagging with category groups (technical, creative, compliance) (verified 2026-05-16; src/review_tag.rs:438 lines ReviewTag hierarchical)

## New Features
- [x] Add a `review_api` module exposing REST endpoints for external tool integration (Slack, Jira, Shotgrid) (verified 2026-05-16; src/review_api.rs:407 lines ReviewApi)
- [x] Implement `review_link` for generating shareable review URLs with expiration and password protection (verified 2026-05-16; src/review_link.rs:641 lines ReviewLink expiration+password)
- [x] Add `review_playlist` module for organizing multiple clips into a sequential review session (verified 2026-05-16; src/review_playlist.rs:236 lines ReviewPlaylist)
- [x] Implement `comparison_mode` with A/B split, onion skin, and difference overlays in `compare` (verified 2026-05-16; src/comparison_mode.rs:426 lines A/B split, onion skin, difference)
- [x] Add `review_snapshot` module for capturing frame grabs with annotations baked in as PNG export (verified 2026-05-16; src/review_snapshot.rs:253 lines ReviewSnapshot)
- [x] Implement `review_automation` for auto-triggering reviews on new version uploads (verified 2026-05-16; src/review_automation.rs:526 lines ReviewAutomation)
- [x] Add `review_permission` fine-grained permissions (can annotate, can approve, can export, can invite) (verified 2026-05-16; src/review_permission.rs:446 lines ReviewPermission)
- [x] Implement `offline_review` for downloading review packages and syncing comments when reconnected (verified 2026-05-16; src/offline_review.rs:653 lines OfflineReview)

## Performance
- [x] Add pagination to `comment` listing for sessions with 1000+ comments (verified 2026-05-16; src/comment.rs:14 paginate_comments, CommentPage, PageRequest)
- [x] Implement delta-based `realtime` sync instead of full-state broadcast for annotation updates (wired 2026-06-05: diff/apply bridge in src/realtime_delta.rs:457 diff_annotations / :510 try_diff_annotations / :571 apply_delta / :644 apply_message + DeltaMessage:369; src/realtime.rs:58 RealtimeEvent::AnnotationDelta variant, :206 broadcast_annotation_delta (incremental, snapshot fallback on serialize error), :264 broadcast_annotation_snapshot — reuses REALTIME_REGISTRY fan-out; 13 tests)
- [x] Cache rendered `drawing` overlays in `compare` to avoid recomputation on scrub (verified-open 2026-05-16: no rendered overlay cache in compare.rs)
- [x] Add lazy loading for `version` history to avoid fetching all versions on session open (verified 2026-05-16; src/version_lazy.rs:139 LazyVersionHistory, paginated provider:79)

## Testing
- [ ] Add tests for `approval_workflow` multi-stage progression with mixed approve/reject decisions
- [ ] Test `realtime` collaboration with simulated concurrent annotation from multiple users
- [ ] Add tests for `export` module generating valid PDF, CSV, and EDL output formats
- [ ] Test `SessionConfigBuilder` with missing required fields and verify graceful defaults
- [ ] Add integration tests for `notify` with mock email and webhook endpoints

## Documentation
- [ ] Document the full review lifecycle: create session -> invite -> annotate -> approve/reject -> export
- [ ] Add workflow diagrams for Simple, MultiStage, Parallel, and Sequential workflow types
- [ ] Document the drawing coordinate system and how annotations map to frame timecodes
- [ ] Add integration guide for connecting review sessions with external project management tools
