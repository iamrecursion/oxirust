//! Wave 2 Slice D shape / value contract tests.
//!
//! These tests do **not** load any real ONNX model. They check two
//! things:
//!   1. Each new pipeline's default configuration matches the contract
//!      documented in its rustdoc.
//!   2. The shared post-processing helpers (`AestheticScore::score`,
//!      `FaceEmbedding::cosine_similarity`) behave correctly on
//!      synthetic inputs so downstream callers can rely on them
//!      regardless of whether a model is available.

mod fixtures;

#[cfg(feature = "aesthetic-score")]
mod aesthetic_contract {
    use oximedia_ml::pipelines::{AestheticImage, AestheticScorerConfig};
    use oximedia_ml::{AestheticScore, MlError};

    #[test]
    fn default_config_is_nima_224_imagenet() {
        let cfg = AestheticScorerConfig::default();
        assert_eq!(cfg.input_size, (224, 224));
        assert!((cfg.mean[0] - 0.485).abs() < 1e-6);
        assert!((cfg.std[1] - 0.224).abs() < 1e-6);
        assert!(cfg.apply_softmax);
    }

    #[test]
    fn aesthetic_image_buffer_validation() {
        let err = AestheticImage::new(vec![0u8; 11], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn score_weighted_mean_uniform_is_5_5() {
        let s = AestheticScore::from_distribution([0.1_f32; 10]);
        assert!((s.score() - 5.5).abs() < 1e-5);
    }

    #[test]
    fn score_weighted_mean_peaked_at_ten_is_ten() {
        let mut dist = [0.0_f32; 10];
        dist[9] = 1.0;
        let s = AestheticScore::from_distribution(dist);
        assert!((s.score() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn score_distribution_accessor_round_trips() {
        let mut dist = [0.0_f32; 10];
        dist[4] = 0.5;
        dist[5] = 0.5;
        let s = AestheticScore::from_distribution(dist);
        assert_eq!(s.distribution(), &dist);
        // sum_{i=1..=10} i * p_{i-1} = 5*0.5 + 6*0.5 = 5.5
        assert!((s.score() - 5.5).abs() < 1e-5);
    }
}

#[cfg(feature = "object-detector")]
mod object_detector_contract {
    use oximedia_ml::pipelines::{
        decode_yolov8_output, DecodeOptions, DetectorImage, ObjectDetectorConfig, YOLOV8_CHANNELS,
        YOLOV8_NUM_CLASSES,
    };
    use oximedia_ml::{BoundingBox, MlError};

    #[test]
    fn default_config_is_yolov8_640_coco() {
        let cfg = ObjectDetectorConfig::default();
        assert_eq!(cfg.input_size, (640, 640));
        assert_eq!(cfg.num_classes, YOLOV8_NUM_CLASSES);
        assert_eq!(YOLOV8_CHANNELS, 84);
        assert!((cfg.conf_threshold - 0.25).abs() < 1e-6);
        assert!((cfg.iou_threshold - 0.45).abs() < 1e-6);
    }

    #[test]
    fn detector_image_buffer_validation() {
        let err = DetectorImage::new(vec![0u8; 11], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn decode_rejects_bad_shape() {
        let data = vec![0.0_f32; 50];
        let shape = vec![1, 10, 5];
        let err =
            decode_yolov8_output(&data, &shape, &DecodeOptions::default()).expect_err("must fail");
        assert!(matches!(err, MlError::Postprocess(_)));
    }

    #[test]
    fn decode_end_to_end_small_cocoa() {
        // 2 classes, 2 anchors, one strong detection + one spurious.
        let n = 2_usize;
        let num_classes = 2_usize;
        let channels = 4 + num_classes;
        let mut data = vec![0.0_f32; channels * n];

        // Anchor 0: strong cat class 1 at (10, 10) 4x4
        data[0] = 10.0;
        data[n] = 10.0;
        data[2 * n] = 4.0;
        data[3 * n] = 4.0;
        data[(4 + 0) * n] = -5.0;
        data[(4 + 1) * n] = 5.0;

        // Anchor 1: below-threshold noise
        data[1] = 50.0;
        data[n + 1] = 50.0;
        data[2 * n + 1] = 4.0;
        data[3 * n + 1] = 4.0;
        data[(4 + 0) * n + 1] = -5.0;
        data[(4 + 1) * n + 1] = -5.0;

        let opts = DecodeOptions {
            num_classes: 2,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        };
        let dets = decode_yolov8_output(&data, &[1, channels, n], &opts).expect("decode ok");
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        let expected_box = BoundingBox::from_xywh_center(10.0, 10.0, 4.0, 4.0);
        assert!((dets[0].bbox.x0 - expected_box.x0).abs() < 1e-5);
        assert!((dets[0].bbox.x1 - expected_box.x1).abs() < 1e-5);
    }
}

#[cfg(feature = "face-embedder")]
mod face_embedder_contract {
    use oximedia_ml::pipelines::{FaceEmbedderConfig, FaceImage};
    use oximedia_ml::{FaceEmbedding, MlError};

    #[test]
    fn default_config_is_arcface_112_512() {
        let cfg = FaceEmbedderConfig::default();
        assert_eq!(cfg.input_size, (112, 112));
        assert_eq!(cfg.embedding_dim, 512);
    }

    #[test]
    fn face_image_buffer_validation() {
        let err = FaceImage::new(vec![0u8; 11], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn face_embedding_identical_cosine_is_one() {
        let emb = FaceEmbedding::from_raw(vec![0.2, 0.5, -0.3, 0.1, 0.4]);
        let sim = emb.cosine_similarity(&emb);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn face_embedding_orthogonal_cosine_is_zero() {
        let mut a = vec![0.0_f32; 512];
        let mut b = vec![0.0_f32; 512];
        a[0] = 1.0;
        b[1] = 1.0;
        let ea = FaceEmbedding::from_raw(a);
        let eb = FaceEmbedding::from_raw(b);
        assert!(ea.cosine_similarity(&eb).abs() < 1e-6);
    }

    #[test]
    fn face_embedding_round_trip_preserves_len() {
        let emb = FaceEmbedding::from_raw(vec![1.0_f32; 128]);
        assert_eq!(emb.len(), 128);
        assert!(!emb.is_empty());
        assert_eq!(emb.as_slice().len(), 128);
    }
}

// Cross-feature sanity: the Detection / AestheticScore / FaceEmbedding
// types are re-exported at the crate root regardless of which
// individual pipeline features are enabled.
#[test]
fn shared_types_available_at_crate_root() {
    use oximedia_ml::{AestheticScore, BoundingBox, Detection, FaceEmbedding};

    let bb = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
    let _det = Detection::new(bb, 0, 0.5);
    let _emb = FaceEmbedding::from_raw(vec![0.0_f32; 4]);
    let _score = AestheticScore::from_distribution([0.1_f32; 10]);
}
