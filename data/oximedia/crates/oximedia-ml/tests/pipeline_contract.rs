//! Integration tests ensuring the typed pipelines honour their contracts
//! regardless of feature configuration.

mod fixtures;

#[cfg(feature = "scene-classifier")]
mod scene {
    use oximedia_ml::pipelines::{SceneClassifierConfig, SceneImage};
    use oximedia_ml::MlError;

    #[test]
    fn scene_image_buffer_validation() {
        let err = SceneImage::new(vec![0u8; 11], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn default_config_is_imagenet_224() {
        let cfg = SceneClassifierConfig::default();
        assert_eq!(cfg.input_size, (224, 224));
        assert_eq!(cfg.top_k, 5);
    }
}

#[cfg(feature = "shot-boundary")]
mod shot {
    use oximedia_ml::pipelines::{ShotBoundaryConfig, ShotBoundaryDetector, ShotFrame};
    use oximedia_ml::TypedPipeline;

    fn solid_frame(w: u32, h: u32, rgb: [u8; 3]) -> ShotFrame {
        let mut buf = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for _ in 0..(w as usize * h as usize) {
            buf.extend_from_slice(&rgb);
        }
        ShotFrame::new(buf, w, h).expect("valid frame")
    }

    #[test]
    fn heuristic_fallback_is_always_available() {
        let det = ShotBoundaryDetector::heuristic(ShotBoundaryConfig::default());
        let out = det.run(Vec::new()).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn heuristic_flags_high_contrast_transition() {
        let det = ShotBoundaryDetector::heuristic(ShotBoundaryConfig {
            threshold: 0.05,
            min_gap: 0,
            ..Default::default()
        });
        let frames = vec![
            solid_frame(48, 27, [0, 0, 0]),
            solid_frame(48, 27, [0, 0, 0]),
            solid_frame(48, 27, [255, 255, 255]),
        ];
        let out = det.run(frames).expect("ok");
        assert!(out.iter().any(|b| b.frame_index == 2));
    }

    #[test]
    fn shot_frame_buffer_validation() {
        let err = ShotFrame::new(vec![0u8; 5], 4, 4).expect_err("must fail");
        assert!(matches!(err, oximedia_ml::MlError::InvalidInput(_)));
    }
}

#[test]
fn feature_flag_smoke_test() {
    // This test always compiles; it simply confirms the crate-root
    // exports are usable without any feature flag.
    use oximedia_ml::{DeviceType, PipelineTask};
    assert!(DeviceType::Cpu.is_available());
    assert_eq!(PipelineTask::Custom as i32, PipelineTask::Custom as i32);
}
