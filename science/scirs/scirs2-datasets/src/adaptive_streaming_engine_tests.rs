use super::*;
use scirs2_core::Array2;
use std::time::Instant;

#[allow(dead_code)]
fn create_test_chunk() -> StreamChunk {
    let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect())
        .expect("Operation failed");
    StreamChunk {
        data,
        timestamp: Instant::now(),
        metadata: ChunkMetadata {
            source_id: "test".to_string(),
            sequence_number: 1,
            characteristics: DataCharacteristics {
                moments: StatisticalMoments {
                    mean: 25.0,
                    variance: 100.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                },
                entropy: 1.0,
                trend: TrendIndicators {
                    linear_slope: 1.0,
                    trend_strength: 0.8,
                    direction: TrendDirection::Increasing,
                    seasonality: 0.2,
                },
                anomaly_score: 0.1,
            },
        },
        quality_score: 0.9,
    }
}

#[test]
fn test_adaptive_engine_creation() {
    let engine = create_adaptive_engine();
    assert!(engine.config.ml_optimization);
}

#[test]
fn test_statistical_moments_calculation() {
    let engine = create_adaptive_engine();
    let data = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )
    .expect("Operation failed");
    let moments = engine.calculate_statistical_moments(&data);
    assert!(moments.is_ok());
    let moments = moments.expect("Operation failed");
    assert!(moments.mean > 0.0);
    assert!(moments.variance >= 0.0);
}
