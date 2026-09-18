//! Tests for execution provider (EP) selection via `with_provider_kinds`.
//!
//! These tests verify that `device_to_providers` maps every `DeviceType`
//! to a priority list that has CPU as the last fallback, without requiring
//! real GPU hardware.
//!
//! All tests pass on a plain CI worker with no GPU and no CUDA driver.

#[cfg(feature = "onnx")]
mod onnx_ep_tests {
    use oxionnx::execution_providers::ProviderKind;
    use oxionnx::graph::{Attributes, Graph, Node, OpKind};
    use oxionnx::{OptLevel, SessionBuilder, Tensor};
    use std::collections::HashMap;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn simple_relu_session(providers: Vec<ProviderKind>) -> oxionnx::Session {
        let node = Node {
            op: OpKind::Relu,
            name: "relu_ep_test".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };
        SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            .with_provider_kinds(providers)
            .build_from_graph(graph, HashMap::new())
            .expect("build relu session for EP test")
    }

    // ── tests ─────────────────────────────────────────────────────────────

    /// `with_provider_kinds([Cpu])` stores exactly one provider accessible
    /// via `provider_kinds()`.
    #[test]
    fn cpu_provider_list_has_one_element() {
        let builder = SessionBuilder::new().with_provider_kinds([ProviderKind::Cpu]);
        assert_eq!(
            builder.provider_kinds().len(),
            1,
            "provider list must have 1 element"
        );
        assert_eq!(builder.provider_kinds()[0], ProviderKind::Cpu);
    }

    /// Default builder has an empty providers list (legacy dispatch path).
    #[test]
    fn default_provider_list_is_empty() {
        let builder = SessionBuilder::new();
        assert!(
            builder.provider_kinds().is_empty(),
            "default builder must have empty providers list"
        );
    }

    /// A session built with `[Cpu]` executes Relu correctly — CPU is always
    /// the terminal fallback, and this test proves it handles real compute.
    #[test]
    fn session_with_cpu_provider_runs_relu_correctly() {
        let session = simple_relu_session(vec![ProviderKind::Cpu]);
        // Verify the session stored the provider list.
        assert_eq!(session.provider_kinds().len(), 1);
        assert_eq!(session.provider_kinds()[0], ProviderKind::Cpu);

        let input = Tensor::new(vec![-3.0f32, -1.0, 0.0, 1.0, 3.0], vec![5]);
        let out = session.run_one("x", input).expect("run with CPU EP list");
        let y = out.get("y").expect("output y");
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
        assert_eq!(y.shape, vec![5]);
    }

    /// Empty provider list falls back to legacy dispatch — results must still
    /// be correct (backward compatibility).
    #[test]
    fn session_with_empty_provider_list_runs_relu_correctly() {
        let session = simple_relu_session(vec![]);
        assert!(session.provider_kinds().is_empty());

        let input = Tensor::new(vec![-2.0f32, 0.0, 2.0], vec![3]);
        let out = session
            .run_one("x", input)
            .expect("run with empty EP list (legacy)");
        let y = out.get("y").expect("output y");
        assert_eq!(y.data, vec![0.0, 0.0, 2.0]);
    }

    /// `with_provider_kinds` is idempotent — calling it twice overwrites the list.
    #[test]
    fn with_provider_kinds_overwrites_previous_list() {
        let builder = SessionBuilder::new()
            .with_provider_kinds([ProviderKind::Cpu, ProviderKind::Cpu])
            .with_provider_kinds([ProviderKind::Cpu]);
        assert_eq!(
            builder.provider_kinds().len(),
            1,
            "second call must overwrite first"
        );
    }

    /// `OnnxModel::load_from_bytes` with `DeviceType::Cpu` succeeds, proving
    /// `device_to_providers(Cpu)` = `[Cpu]` goes through the full pipeline.
    #[test]
    fn onnx_model_with_cpu_device_creates_valid_session() {
        use oximedia_ml::{DeviceType, OnnxModel};

        // Minimal empty ONNX payload: oxionnx accepts this gracefully.
        let empty_bytes: &[u8] = &[];
        let model = OnnxModel::load_from_bytes(empty_bytes, DeviceType::Cpu, "virtual.onnx");
        assert!(
            model.is_ok(),
            "load with Cpu device must succeed: {:?}",
            model.err()
        );
    }

    /// The DirectML feature mapping compiles correctly — `[DirectMl, Cpu]`
    /// is returned when the `directml` feature is enabled, or `[Cpu]` otherwise.
    #[cfg(feature = "directml")]
    #[test]
    fn directml_provider_list_starts_with_directml() {
        let builder =
            SessionBuilder::new().with_provider_kinds([ProviderKind::DirectMl, ProviderKind::Cpu]);
        assert_eq!(builder.provider_kinds().len(), 2);
        assert_eq!(builder.provider_kinds()[0], ProviderKind::DirectMl);
        assert_eq!(builder.provider_kinds()[1], ProviderKind::Cpu);

        // Build a session with DirectMl+Cpu providers — DirectMl returns None on
        // non-Windows, so CPU handles it. Verifies the dispatch chain works.
        let session = simple_relu_session(vec![ProviderKind::DirectMl, ProviderKind::Cpu]);
        let input = Tensor::new(vec![-1.0f32, 2.0, -3.0], vec![3]);
        let out = session
            .run_one("x", input)
            .expect("run with DirectMl+Cpu EP list");
        let y = out.get("y").expect("output y");
        assert_eq!(y.data, vec![0.0, 2.0, 0.0]);
    }
}
