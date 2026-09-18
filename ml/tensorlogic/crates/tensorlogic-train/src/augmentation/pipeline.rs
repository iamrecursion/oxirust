use scirs2_core::ndarray::ArrayD;

use super::error::AugmentationError;
use super::functional::{
    center_crop_2d, clip, dropout, gaussian_noise, normalize, random_crop_2d, random_hflip,
    random_vflip,
};
use super::rng::AugRng;

/// A single step in an augmentation pipeline.
#[derive(Debug, Clone)]
pub enum AugmentationStep {
    GaussianNoise { std: f64 },
    Dropout { p: f64 },
    RandomHFlip { p: f64 },
    RandomVFlip { p: f64 },
    RandomCrop { crop_h: usize, crop_w: usize },
    CenterCrop { crop_h: usize, crop_w: usize },
    Normalize { mean: Vec<f64>, std: Vec<f64> },
    Clip { min_val: f64, max_val: f64 },
}

/// A composable, ordered sequence of augmentation steps.
///
/// Steps are applied left-to-right. Each step receives its own `AugRng`
/// derived from the pipeline seed advanced by the step index so results
/// are deterministic given the same seed.
#[derive(Debug, Clone)]
pub struct AugmentationPipeline {
    /// The ordered list of augmentation steps.
    pub steps: Vec<AugmentationStep>,
    /// Seed used to derive per-step RNG states.
    pub rng_seed: u64,
}

impl AugmentationPipeline {
    /// Create an empty pipeline with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            steps: Vec::new(),
            rng_seed: seed,
        }
    }

    /// Append a step and return `self` (builder pattern).
    pub fn add_step(mut self, step: AugmentationStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Apply all steps to `input` sequentially.
    ///
    /// A fresh `AugRng` is derived for each step from `rng_seed ^ (step_index * prime)`,
    /// guaranteeing reproducibility while avoiding correlation between steps.
    pub fn apply(
        &self,
        input: &ArrayD<f64>,
        training: bool,
    ) -> Result<ArrayD<f64>, AugmentationError> {
        let mut current = input.clone();
        for (i, step) in self.steps.iter().enumerate() {
            // Derive a per-step seed.
            let step_seed = self
                .rng_seed
                .wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut rng = AugRng::new(step_seed);

            current = match step {
                AugmentationStep::GaussianNoise { std } => {
                    gaussian_noise(&current, *std, &mut rng)?
                }
                AugmentationStep::Dropout { p } => dropout(&current, *p, training, &mut rng)?,
                AugmentationStep::RandomHFlip { p } => random_hflip(&current, *p, &mut rng)?,
                AugmentationStep::RandomVFlip { p } => random_vflip(&current, *p, &mut rng)?,
                AugmentationStep::RandomCrop { crop_h, crop_w } => {
                    random_crop_2d(&current, *crop_h, *crop_w, &mut rng)?
                }
                AugmentationStep::CenterCrop { crop_h, crop_w } => {
                    center_crop_2d(&current, *crop_h, *crop_w)?
                }
                AugmentationStep::Normalize { mean, std } => normalize(&current, mean, std)?,
                AugmentationStep::Clip { min_val, max_val } => clip(&current, *min_val, *max_val),
            };
        }
        Ok(current)
    }

    /// Number of steps in the pipeline.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

/// Statistics comparing original and augmented data.
#[derive(Debug, Clone)]
pub struct AugStats {
    /// Mean of the original array.
    pub original_mean: f64,
    /// Standard deviation of the original array.
    pub original_std: f64,
    /// Mean of the augmented array.
    pub augmented_mean: f64,
    /// Standard deviation of the augmented array.
    pub augmented_std: f64,
    /// Fraction of elements whose value changed (|orig − aug| > ε).
    pub element_change_ratio: f64,
}

impl AugStats {
    /// Compute statistics comparing `original` and `augmented`.
    pub fn compute(original: &ArrayD<f64>, augmented: &ArrayD<f64>) -> Self {
        let orig_flat: Vec<f64> = original.iter().copied().collect();
        let aug_flat: Vec<f64> = augmented.iter().copied().collect();
        let n = orig_flat.len().max(1);

        let orig_mean = orig_flat.iter().sum::<f64>() / n as f64;
        let aug_mean = aug_flat.iter().sum::<f64>() / aug_flat.len().max(1) as f64;

        let orig_var = orig_flat
            .iter()
            .map(|&x| (x - orig_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let aug_var = aug_flat
            .iter()
            .map(|&x| (x - aug_mean).powi(2))
            .sum::<f64>()
            / aug_flat.len().max(1) as f64;

        let compare_n = orig_flat.len().min(aug_flat.len()).max(1);
        let changed = orig_flat
            .iter()
            .zip(aug_flat.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-12)
            .count();

        AugStats {
            original_mean: orig_mean,
            original_std: orig_var.sqrt(),
            augmented_mean: aug_mean,
            augmented_std: aug_var.sqrt(),
            element_change_ratio: changed as f64 / compare_n as f64,
        }
    }

    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "orig μ={:.4} σ={:.4} | aug μ={:.4} σ={:.4} | changed {:.1}%",
            self.original_mean,
            self.original_std,
            self.augmented_mean,
            self.augmented_std,
            self.element_change_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod aug_tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    fn make_rng() -> AugRng {
        AugRng::new(0xDEAD_BEEF)
    }

    fn ones(shape: &[usize]) -> ArrayD<f64> {
        use scirs2_core::ndarray::IxDyn;
        let n: usize = shape.iter().product();
        ArrayD::from_shape_vec(IxDyn(shape), vec![1.0f64; n]).expect("shape ok")
    }

    fn arange(shape: &[usize]) -> ArrayD<f64> {
        use scirs2_core::ndarray::IxDyn;
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape ok")
    }

    // ---- gaussian_noise ----

    #[test]
    fn test_gaussian_noise_shape_preserved() {
        let input = ones(&[3, 4]);
        let mut rng = make_rng();
        let out = gaussian_noise(&input, 0.1, &mut rng).expect("ok");
        assert_eq!(out.shape(), input.shape());
    }

    #[test]
    fn test_gaussian_noise_mean_near_original() {
        // With std=0.01 and 1000 elements the mean should stay close to 1.0.
        let input = ones(&[10, 100]);
        let mut rng = make_rng();
        let out = gaussian_noise(&input, 0.01, &mut rng).expect("ok");
        let sum: f64 = out.iter().sum();
        let mean = sum / 1000.0;
        assert!((mean - 1.0).abs() < 0.05, "mean {mean} too far from 1.0");
    }

    // ---- dropout ----

    #[test]
    fn test_dropout_training_zeroes_some() {
        let input = ones(&[100]);
        let mut rng = make_rng();
        let out = dropout(&input, 0.5, true, &mut rng).expect("ok");
        let zero_count = out.iter().filter(|&&x| x == 0.0).count();
        // With p=0.5 and 100 elements, expect some zeros.
        assert!(zero_count > 0, "expected some zeros");
        assert!(zero_count < 100, "not all should be zero");
    }

    #[test]
    fn test_dropout_inference_unchanged() {
        let input = arange(&[5, 5]);
        let mut rng = make_rng();
        let out = dropout(&input, 0.9, false, &mut rng).expect("ok");
        assert_eq!(out, input);
    }

    // ---- dropout_mask ----

    #[test]
    fn test_dropout_mask_shape() {
        use super::super::functional::dropout_mask;
        let mut rng = make_rng();
        let mask = dropout_mask(&[4, 4], 0.3, &mut rng).expect("ok");
        assert_eq!(mask.shape(), &[4, 4]);
        for &v in mask.iter() {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    // ---- mixup ----

    #[test]
    fn test_mixup_shape() {
        use super::super::functional::mixup;
        let x1 = ones(&[3, 4]);
        let x2 = arange(&[3, 4]);
        let mut rng = make_rng();
        let (mixed, _lam) = mixup(&x1, &x2, 1.0, &mut rng).expect("ok");
        assert_eq!(mixed.shape(), x1.shape());
    }

    #[test]
    fn test_mixup_lambda_range() {
        use super::super::functional::mixup;
        let x1 = ones(&[2, 2]);
        let x2 = ones(&[2, 2]);
        let mut rng = make_rng();
        for _ in 0..50 {
            let (_mixed, lam) = mixup(&x1, &x2, 1.0, &mut rng).expect("ok");
            assert!((0.0..=1.0).contains(&lam), "lambda={lam} out of range");
        }
    }

    // ---- cutmix ----

    #[test]
    fn test_cutmix_shape() {
        use super::super::functional::cutmix;
        let x1 = ones(&[1, 3, 8, 8]);
        let x2 = arange(&[1, 3, 8, 8]);
        let mut rng = make_rng();
        let (mixed, _lam) = cutmix(&x1, &x2, 1.0, &mut rng).expect("ok");
        assert_eq!(mixed.shape(), x1.shape());
    }

    #[test]
    fn test_cutmix_lambda_range() {
        use super::super::functional::cutmix;
        let x1 = ones(&[1, 4, 8, 8]);
        let x2 = arange(&[1, 4, 8, 8]);
        let mut rng = make_rng();
        for _ in 0..20 {
            let (_mixed, lam) = cutmix(&x1, &x2, 1.0, &mut rng).expect("ok");
            assert!((0.0..=1.0).contains(&lam), "lambda={lam} out of range");
        }
    }

    // ---- random_crop_2d ----

    #[test]
    fn test_random_crop_2d_shape() {
        let input = arange(&[3, 16, 16]);
        let mut rng = make_rng();
        let out = random_crop_2d(&input, 12, 12, &mut rng).expect("ok");
        assert_eq!(out.shape(), &[3, 12, 12]);
    }

    #[test]
    fn test_random_crop_invalid_size() {
        let input = ones(&[8, 8]);
        let mut rng = make_rng();
        let result = random_crop_2d(&input, 16, 8, &mut rng);
        assert!(result.is_err(), "crop larger than input should fail");
    }

    // ---- center_crop_2d ----

    #[test]
    fn test_center_crop_2d_shape() {
        let input = arange(&[1, 3, 32, 32]);
        let out = center_crop_2d(&input, 24, 24).expect("ok");
        assert_eq!(out.shape(), &[1, 3, 24, 24]);
    }

    // ---- random_hflip ----

    #[test]
    fn test_random_hflip_probability_zero() {
        let input = arange(&[2, 4, 4]);
        let mut rng = make_rng();
        let out = random_hflip(&input, 0.0, &mut rng).expect("ok");
        assert_eq!(out, input, "p=0 must leave input unchanged");
    }

    #[test]
    fn test_random_hflip_probability_one() {
        // Flip of flip should give back original.
        let input = arange(&[1, 4, 4]);
        let mut rng = make_rng();
        let flipped = random_hflip(&input, 1.0, &mut rng).expect("ok");
        assert_ne!(flipped, input, "p=1 must flip");
        let mut rng2 = make_rng();
        let double_flipped = random_hflip(&flipped, 1.0, &mut rng2).expect("ok");
        assert_eq!(double_flipped, input, "double flip = identity");
    }

    // ---- normalize / denormalize ----

    #[test]
    fn test_normalize_and_denormalize_roundtrip() {
        use super::super::functional::denormalize;
        let input = arange(&[2, 3, 4, 4]);
        let mean = vec![0.485, 0.456, 0.406];
        let std = vec![0.229, 0.224, 0.225];

        let normed = normalize(&input, &mean, &std).expect("normalize ok");
        let restored = denormalize(&normed, &mean, &std).expect("denormalize ok");

        for (a, b) in input.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-9, "roundtrip mismatch: {a} vs {b}");
        }
    }

    // ---- clip ----

    #[test]
    fn test_clip_bounds() {
        let input = arange(&[10]);
        let clipped = clip(&input, 2.0, 7.0);
        for &v in clipped.iter() {
            assert!((2.0..=7.0).contains(&v), "value {v} out of clipped range");
        }
    }

    // ---- pipeline ----

    #[test]
    fn test_pipeline_apply_empty() {
        let pipeline = AugmentationPipeline::new(42);
        let input = arange(&[4, 4]);
        let out = pipeline.apply(&input, true).expect("ok");
        assert_eq!(out, input, "empty pipeline is identity");
    }

    #[test]
    fn test_pipeline_apply_noise_step() {
        let pipeline = AugmentationPipeline::new(99)
            .add_step(AugmentationStep::GaussianNoise { std: 0.01 })
            .add_step(AugmentationStep::Clip {
                min_val: -10.0,
                max_val: 100.0,
            });
        let input = ones(&[20, 20]);
        let out = pipeline.apply(&input, true).expect("ok");
        assert_eq!(out.shape(), input.shape());
    }

    // ---- AugStats ----

    #[test]
    fn test_aug_stats_compute() {
        let orig = ones(&[10]);
        let aug = arange(&[10]);
        let stats = AugStats::compute(&orig, &aug);
        assert!((stats.original_mean - 1.0).abs() < 1e-9);
        // At least some elements changed.
        assert!(stats.element_change_ratio > 0.0);
    }

    #[test]
    fn test_aug_stats_summary_nonempty() {
        let orig = ones(&[5]);
        let aug = arange(&[5]);
        let stats = AugStats::compute(&orig, &aug);
        let summary = stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("μ"));
    }
}
