/// Errors that can occur during augmentation operations.
#[derive(Debug, Clone)]
pub enum AugmentationError {
    /// Probability value is outside [0, 1].
    InvalidProbability(f64),
    /// Alpha parameter is non-positive.
    InvalidAlpha(f64),
    /// Two arrays have incompatible shapes.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Input array has zero elements.
    EmptyInput,
    /// Noise std is negative.
    InvalidNoise { std: f64 },
    /// Crop size exceeds input size along that dimension.
    InvalidCrop { crop_size: usize, input_size: usize },
}

impl std::fmt::Display for AugmentationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AugmentationError::InvalidProbability(p) => {
                write!(f, "probability {p} is outside [0, 1]")
            }
            AugmentationError::InvalidAlpha(a) => {
                write!(f, "alpha {a} must be positive")
            }
            AugmentationError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            AugmentationError::EmptyInput => write!(f, "input array is empty"),
            AugmentationError::InvalidNoise { std } => {
                write!(f, "noise std {std} must be non-negative")
            }
            AugmentationError::InvalidCrop {
                crop_size,
                input_size,
            } => {
                write!(f, "crop size {crop_size} exceeds input size {input_size}")
            }
        }
    }
}

impl std::error::Error for AugmentationError {}
