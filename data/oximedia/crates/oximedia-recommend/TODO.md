# oximedia-recommend TODO

## Current Status
- 37 modules for content recommendation and discovery engine
- Key types: RecommendationEngine, RecommendationRequest, Recommendation, RecommendationResults
- Strategies: ContentBased, Collaborative, Hybrid, Personalized, Trending
- Core modules: collaborative (matrix factorization), content (similarity), hybrid (combine), profile (user), history (track), rating (explicit/implicit), trending (detect), personalize (engine), diversity (ensure), freshness (balance), rank (score), explain (generate)
- Advanced modules: ab_test, als (Alternating Least Squares), bandits, calibration, cold_start, collab_filter, content_based, context_signal, decay_model, dense_linalg, exploration_policy, feature_store, feedback_signal, impression_tracker, item_similarity, popularity_bias, recommendation_score, score_cache, sequence_model, session, svd_pp (SVD++), user_profile
- Dependencies: oximedia-core, oximedia-search, rayon, chrono, uuid, serde

## Enhancements
- [x] Add real-time model updating in `collaborative::matrix::CollaborativeEngine` (incremental matrix factorization) (verified 2026-05-16; src/collaborative/matrix.rs:174 IncrementalMF config, SGD update:198)
- [x] Implement user segment-based recommendations in `personalize` (cluster users into segments) (verified 2026-05-16; src/personalize/segments.rs:21 UserSegment struct, UserSegmenter)
- [x] Extend `diversity::ensure::DiversityEnforcer` with maximal marginal relevance (MMR) reranking (verified 2026-05-16; src/diversity/ensure.rs:9 MmrReranker, src/diversity/deduplication.rs:64 MaximalMarginalRelevance)
- [x] Add `impression_tracker` deduplication — never recommend already-seen content
- [x] Implement `cold_start` with popularity-based fallback and demographic-based initialization (verified 2026-06-01; src/cold_start.rs:209 `resolve_popularity`, :230 `resolve_demographic`, :303 `resolve_hybrid`)
- [x] Extend `explain::generate` with visual explanation data (feature importance scores) (verified 2026-06-01; src/explain/generate.rs: `ImportanceDirection`, `FeatureImportance`, `decompose_reasons`, `DetailedExplanationGenerator::generate_with_importance`; 4 tests added)
- [x] Add `ab_test` statistical significance testing (chi-squared, t-test for engagement metrics)
- [x] Implement rate limiting in `RecommendationEngine` to prevent abuse of recommendation API

## New Features
- [x] Add knowledge graph-based recommendations — leverage media metadata relationships (director, genre, cast) (verified 2026-05-16; src/knowledge_graph.rs:697 lines)
- [x] Implement session-based recommendations in `session` module (short-term user intent modeling) (verified 2026-05-16; src/session_recommend.rs:630 lines, src/session/mod.rs)
- [x] Add multi-objective optimization — balance engagement, diversity, and freshness simultaneously (verified 2026-05-16; src/multi_objective.rs:536 lines)
- [x] Implement federated learning support — train collaborative models without centralizing user data (verified 2026-05-16; src/federated.rs:866 lines)
- [x] Add content embargo/scheduling — time-gate recommendations for release-date-aware content (verified 2026-05-16; src/embargo.rs:484 lines)
- [x] Implement cross-domain recommendations (recommend audio content to video watchers based on shared interests) (verified 2026-05-16; src/cross_domain.rs:705 lines)
- [x] Add recommendation fairness metrics — measure and enforce exposure equity across content creators (verified 2026-05-16; src/fairness.rs:618 lines)
- [x] Implement contextual bandits in `bandits` for exploration/exploitation in live recommendations (verified 2026-05-16; src/contextual_bandits.rs:534 lines, src/bandits/mod.rs)

## Performance
- [x] Implement approximate nearest neighbor search in `item_similarity` using locality-sensitive hashing (verified 2026-05-16; src/lsh.rs:465 lines)
- [x] Add `score_cache` with LRU eviction and TTL-based invalidation for hot recommendation paths (verified 2026-05-16; src/score_cache.rs:435 lines)
- [x] Optimize `dense_linalg` matrix operations using blocked algorithms for cache efficiency (verified 2026-06-01; src/dense_linalg.rs: `TILE=64`, `DenseMatrix::matmul` (cache-blocked GEMM), `DenseMatrix::matmul_parallel` (rayon row-block parallel); 5 tests added)
- [x] Pre-compute user embeddings in `collaborative` and cache for sub-millisecond recommendation serving (verified 2026-06-01; src/collaborative/matrix.rs: `user_embedding_cache: Option<HashMap<usize, Vec<f32>>>` in `LatentFactorModel`, `precompute_user_embeddings`, `recommend_precomputed`, `recommend_on_demand`; cache cleared on `retrain`; 3 tests added)
- [x] Parallelize `RecommendationEngine::recommend()` strategy evaluation using rayon (verified 2026-06-01; src/lib.rs:462 `get_hybrid_parallel`, :558 `use rayon::prelude::*`, :571 `par_iter()` across strategies)
- [x] Implement batch recommendation generation for offline/pre-computation scenarios (verified 2026-05-16; src/batch_recommend.rs:768 lines)

## Testing
- [x] Add offline evaluation tests using precision@k, recall@k, and NDCG metrics (verified 2026-06-01; src/evaluation.rs all three metrics with tests at :287-496)
- [x] Test `cold_start` behavior for brand-new users with zero interaction history (verified 2026-06-01; src/cold_start.rs:433 `test_is_cold_user`, :541 `test_empty_resolver_returns_empty`)
- [x] Add diversity measurement tests — verify `DiversityEnforcer` actually increases category spread (verified 2026-06-01; src/diversity/ensure.rs:383 `test_enforce_diversity_caps_category`, :428/:436 diversity-score tests)
- [x] Test `trending::detect` with synthetic view spikes and verify detection latency (verified 2026-06-01; src/trending_detection.rs:332 `test_spike_triggers_viral`, :315/:353 companion tests)
- [x] Add regression tests for `svd_pp` and `als` with small known-answer datasets (verified 2026-06-01; src/als.rs:762 `test_als_predict_known_pair`; src/svd_pp.rs:553 `test_svdpp_predict_known_pair`)

## Documentation
- [ ] Add recommendation algorithm selection guide (when to use content-based vs collaborative vs hybrid)
- [ ] Document A/B testing workflow with metric collection and analysis
- [ ] Add integration guide for connecting RecommendationEngine to a media platform backend
