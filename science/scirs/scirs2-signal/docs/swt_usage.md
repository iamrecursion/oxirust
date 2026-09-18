# Stationary Wavelet Transform (SWT)

## Introduction

The Stationary Wavelet Transform (SWT), also known as the Undecimated Wavelet Transform or the à trous algorithm, is a translation-invariant version of the Discrete Wavelet Transform (DWT). Unlike the standard DWT, the SWT does not downsample the signal after filtering, which makes it shift-invariant. This property is particularly useful for applications like denoising, feature extraction, and pattern recognition.

## Main Differences Between SWT and DWT

1. **No Downsampling**: The SWT does not perform downsampling after filtering, which results in coefficient arrays of the same length as the input signal at each level.

2. **Translation Invariance**: The SWT is translation-invariant, meaning that a small shift in the input signal will cause a corresponding shift in the transform coefficients, without aliasing effects.

3. **Redundancy**: SWT is a redundant transform, providing more coefficients than the original signal, which can be beneficial for certain applications.

4. **Filter Upsampling**: Instead of downsampling the signal, the SWT upsamples the filters at each level by inserting zeros between the filter coefficients.

## SWT Implementation in scirs2-signal

The `scirs2-signal` crate provides a comprehensive implementation of the SWT with the following functions:

### Single-Level Decomposition and Reconstruction

- `swt_decompose`: Performs one level of the stationary wavelet transform
- `swt_reconstruct`: Performs one level of the inverse stationary wavelet transform

### Multi-Level Decomposition and Reconstruction

- `swt`: Performs a multi-level stationary wavelet transform
- `iswt`: Performs a multi-level inverse stationary wavelet transform

## Example Usage

### Basic Decomposition and Reconstruction

```rust
use scirs2_signal::swt::{swt_decompose, swt_reconstruct};
use scirs2_signal::dwt::Wavelet;

// Generate a simple signal
let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Perform SWT using the Haar wavelet at level 1
let (ca, cd) = swt_decompose(&signal, Wavelet::Haar, 1, None).unwrap();

// Reconstruct the signal
let reconstructed = swt_reconstruct(&ca, &cd, Wavelet::Haar, 1).unwrap();

// Verify that the reconstruction matches the original
for (x, y) in signal.iter().zip(reconstructed.iter()) {
    assert!((x - y).abs() < 1e-10);
}
```

### Multi-Level Transform

```rust
use scirs2_signal::swt::{swt, iswt};
use scirs2_signal::dwt::Wavelet;

// Generate a simple signal
let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Perform multi-level SWT using the Daubechies wavelet (3 levels)
let (details, approx) = swt(&signal, Wavelet::DB(4), 3, None).unwrap();

// details[0] contains level 1 detail coefficients
// details[1] contains level 2 detail coefficients
// details[2] contains level 3 detail coefficients
// approx contains level 3 approximation coefficients

// Reconstruct the signal
let reconstructed = iswt(&details, &approx, Wavelet::DB(4)).unwrap();
```

### Signal Denoising

```rust
use scirs2_signal::swt::{swt, iswt};
use scirs2_signal::dwt::Wavelet;

// Assume we have a noisy signal
let noisy_signal = vec![/* ... */];

// Perform SWT decomposition
let (details, approx) = swt(&noisy_signal, Wavelet::DB(4), 3, None).unwrap();

// Apply thresholding to detail coefficients
let mut modified_details = details.clone();
for level in 0..details.len() {
    let threshold = 0.1 / (level + 1) as f64; // Threshold value (adjust as needed)
    for val in &mut modified_details[level] {
        if val.abs() < threshold {
            *val = 0.0; // Hard thresholding
        }
    }
}

// Reconstruct the denoised signal
let denoised_signal = iswt(&modified_details, &approx, Wavelet::DB(4)).unwrap();
```

## Advanced Features

### Extension Modes

The SWT implementation supports different signal extension modes:

- `"symmetric"`: Signal is extended by mirroring (default)
- `"periodic"`: Signal is extended by wrapping around
- `"zero"`: Signal is extended with zeros

```rust
// Using symmetric extension mode (default)
let (ca, cd) = swt_decompose(&signal, Wavelet::Haar, 1, Some("symmetric")).unwrap();

// Using periodic extension
let (ca, cd) = swt_decompose(&signal, Wavelet::Haar, 1, Some("periodic")).unwrap();

// Using zero-padding
let (ca, cd) = swt_decompose(&signal, Wavelet::Haar, 1, Some("zero")).unwrap();
```

### Wavelet Types

The SWT supports all the same wavelet types as the DWT:

- `Haar`: Haar wavelet (equivalent to DB1)
- `DB(n)`: Daubechies wavelets with n vanishing moments (1-20)
- `Sym(n)`: Symlet wavelets with n vanishing moments (2-20)
- `Coif(n)`: Coiflet wavelets with n vanishing moments (1-5)
- `BiorNrNd`: Biorthogonal wavelets with Nr/Nd vanishing moments
- `RBioNrNd`: Reverse biorthogonal wavelets with Nr/Nd vanishing moments

## Performance Considerations

- SWT is more computationally intensive than DWT due to its redundant nature
- Memory usage increases linearly with the number of decomposition levels
- For multi-level transforms, each coefficient array has the same size as the input signal

## Applications

The Stationary Wavelet Transform is particularly useful for:

1. **Denoising**: Often provides better results than DWT due to its shift-invariance
2. **Feature extraction**: Consistent features regardless of signal positioning
3. **Edge detection**: Better localization of edges in signals and images
4. **Change point detection**: Identifying abrupt changes in signals
5. **Time-frequency analysis**: Analyzing non-stationary signals

## References

1. M. Holschneider, R. Kronland-Martinet, J. Morlet, and Ph. Tchamitchian, "A real-time algorithm for signal analysis with the help of the wavelet transform," in Wavelets: Time-Frequency Methods and Phase Space, J.M. Combes, A. Grossmann, and Ph. Tchamitchian, Eds. Berlin: Springer-Verlag, 1989, pp. 286-297.

2. M.J. Shensa, "The discrete wavelet transform: Wedding the à trous and Mallat algorithms," IEEE Trans. Signal Processing, vol. 40, pp. 2464-2482, 1992.

3. G.P. Nason and B.W. Silverman, "The stationary wavelet transform and some statistical applications," in Wavelets and Statistics, A. Antoniadis and G. Oppenheim, Eds. New York: Springer-Verlag, 1995, pp. 281-299.