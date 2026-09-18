# Wavelet Families in scirs2-signal

This document provides an overview of the wavelet families implemented in the scirs2-signal crate, including their properties, use cases, and examples.

## Table of Contents

1. [Introduction to Wavelets](#introduction-to-wavelets)
2. [Available Wavelet Families](#available-wavelet-families)
3. [Wavelet Selection Guide](#wavelet-selection-guide)
4. [Wavelet Transform Types](#wavelet-transform-types)
5. [Performance Considerations](#performance-considerations)
6. [Examples](#examples)

## Introduction to Wavelets

Wavelets are mathematical functions that decompose a signal into different frequency components, each with a resolution matched to its scale. Unlike the Fourier transform, which uses sine and cosine functions, wavelets are localized in both time and frequency domains, making them well-suited for analyzing non-stationary signals or signals with discontinuities.

Key wavelet properties include:

- **Vanishing moments**: Higher vanishing moments provide better approximation of smooth functions
- **Support size**: The length of the filter, affecting computational complexity
- **Regularity**: Smoothness of the wavelet function
- **Symmetry**: Important for preventing phase distortion

## Available Wavelet Families

### Haar Wavelet

The simplest wavelet, with a rectangular shape.

- **Properties**: 
  - Discontinuous step function
  - Compact support
  - One vanishing moment
  - Orthogonal
  - Symmetrical
  
- **Use cases**: 
  - Edge detection
  - Simple data compression
  - Fast computation
  - Signals with sharp transitions

- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
  let (approx, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();
  ```

### Daubechies Wavelets (DB)

A family of orthogonal wavelets with varying numbers of vanishing moments.

- **Properties**:
  - Compact support (length 2N for dbN)
  - N vanishing moments
  - Asymmetrical (except for db1/Haar)
  - Orthogonal
  - Good time-frequency localization
  
- **Use cases**:
  - Signal and image compression
  - Noise reduction
  - Feature extraction
  - Fractal analysis
  
- **Available orders**: DB1 (same as Haar) through DB20
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  // DB4 has 4 vanishing moments and filter length 8
  let (approx, detail) = dwt_decompose(&signal, Wavelet::DB(4), None).unwrap();
  ```

### Symlets (Sym)

Modified version of Daubechies wavelets with increased symmetry.

- **Properties**:
  - Nearly symmetrical
  - Compact support
  - N vanishing moments for symN
  - Orthogonal
  
- **Use cases**:
  - Applications requiring phase information
  - Signal processing where symmetry is important
  - Image processing with less distortion
  
- **Available orders**: Sym2 through Sym20
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  // Sym8 has 8 vanishing moments
  let (approx, detail) = dwt_decompose(&signal, Wavelet::Sym(8), None).unwrap();
  ```

### Coiflets (Coif)

Designed by Daubechies at the request of Ronald Coifman, with higher symmetry than Daubechies wavelets.

- **Properties**:
  - More symmetrical than Daubechies
  - 2N vanishing moments for scaling function, 2N-1 for wavelet function
  - Compact support
  - Orthogonal
  
- **Use cases**:
  - Approximating smooth functions
  - Data compression with better reconstruction
  - Signal analysis requiring symmetry
  
- **Available orders**: Coif1 through Coif5
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  // Coif3 has 6 vanishing moments for the scaling function
  let (approx, detail) = dwt_decompose(&signal, Wavelet::Coif(3), None).unwrap();
  ```

### Biorthogonal Wavelets (Bior)

Wavelets that allow for symmetric filters with compact support by relaxing orthogonality.

- **Properties**:
  - Perfect reconstruction
  - Symmetrical
  - Different vanishing moments for decomposition and reconstruction
  - Compact support
  
- **Use cases**:
  - Image processing (JPEG2000 standard)
  - Applications requiring symmetry
  - Progressive data transmission
  
- **Available combinations**: Specified by two numbers Nr.Nd (e.g., Bior3.5)
  - Nr: Number of vanishing moments for reconstruction
  - Nd: Number of vanishing moments for decomposition
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  // Bior3.5 has 3 vanishing moments for reconstruction,
  // 5 for decomposition
  let (approx, detail) = dwt_decompose(
      &signal, 
      Wavelet::BiorNrNd { nr: 3, nd: 5 }, 
      None
  ).unwrap();
  ```

### Reverse Biorthogonal Wavelets (RBio)

Reverse of biorthogonal wavelets, swapping the decomposition and reconstruction filters.

- **Properties**:
  - Same as biorthogonal, but with reversed roles for decomposition and reconstruction
  
- **Use cases**:
  - Same as biorthogonal, but may provide better results in specific applications
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  let (approx, detail) = dwt_decompose(
      &signal, 
      Wavelet::RBioNrNd { nr: 3, nd: 5 }, 
      None
  ).unwrap();
  ```

### Meyer Wavelet

Infinitely differentiable wavelets with infinite support, defined in the frequency domain.

- **Properties**:
  - Defined in the frequency domain
  - Infinite support in time domain
  - FIR approximation used for implementation
  - Symmetrical
  - Good localization in frequency domain
  
- **Use cases**:
  - Harmonic analysis
  - Applications requiring excellent frequency localization
  - Theoretical analysis
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  let (approx, detail) = dwt_decompose(&signal, Wavelet::Meyer, None).unwrap();
  ```

### Discrete Meyer Wavelet (DMeyer)

A finite impulse response (FIR) approximation of the Meyer wavelet.

- **Properties**:
  - FIR approximation of Meyer wavelet
  - Compact support (implemented with 62 filter taps)
  - Good frequency localization
  - More computationally efficient than the continuous Meyer wavelet
  
- **Use cases**:
  - Applications requiring Meyer-like properties with better computational efficiency
  - Signal denoising
  - Feature extraction
  
- **Code example**:
  ```rust
  use scirs2_signal::dwt::{dwt_decompose, Wavelet};
  
  let (approx, detail) = dwt_decompose(&signal, Wavelet::DMeyer, None).unwrap();
  ```

## Available Complex Wavelets for Continuous Wavelet Transform (CWT)

In addition to the discrete wavelet families above (used primarily with DWT), the library also provides several complex wavelets for use with the Continuous Wavelet Transform (CWT). These are particularly useful for analyzing signals where phase information is important.

### Complex Morlet Wavelet

A complex exponential modulated by a Gaussian function with additional parameters for controlling the shape.

- **Properties**:
  - Excellent time-frequency localization
  - Adjustable center frequency and bandwidth
  - Optional asymmetry parameter
  - Complex-valued
  
- **Use cases**:
  - General purpose time-frequency analysis
  - Analyzing non-stationary signals
  - Phase information extraction
  - Ridge detection in time-frequency plane
  
- **Code example**:
  ```rust
  use scirs2_signal::wavelets::{cwt, complex_morlet};
  
  // Define signal and scales
  let scales = vec![1.0, 2.0, 4.0, 8.0, 16.0];
  
  // CWT with complex Morlet, center_frequency=5.0, bandwidth=1.0, symmetry=0.0
  let result = cwt(
      &signal,
      |points, scale| complex_morlet(points, 5.0, 1.0, 0.0, scale),
      &scales
  ).unwrap();
  ```

### Complex Gaussian Wavelet

A complex-valued derivative of the Gaussian function, with the order parameter controlling the number of oscillations.

- **Properties**:
  - Derivatives of Gaussian function
  - Adjustable oscillations via order parameter
  - Complex-valued
  - Good time localization
  
- **Use cases**:
  - Edge detection
  - Singularity detection
  - Feature extraction
  - Applications requiring good time localization
  
- **Code example**:
  ```rust
  use scirs2_signal::wavelets::{cwt, complex_gaussian};
  
  // CWT with Complex Gaussian of order 4
  let result = cwt(
      &signal,
      |points, scale| complex_gaussian(points, 4, scale),
      &scales
  ).unwrap();
  ```

### Shannon Wavelet

A band-limited wavelet defined using the sinc function modulated by a complex exponential.

- **Properties**:
  - Excellent frequency localization
  - Sharp frequency cutoff
  - Poor time localization
  - Complex-valued
  
- **Use cases**:
  - Narrow-band frequency analysis
  - Spectral decomposition
  - Applications where frequency precision is critical
  
- **Code example**:
  ```rust
  use scirs2_signal::wavelets::{cwt, shannon};
  
  // CWT with Shannon wavelet, center_frequency=1.0, bandwidth=0.5
  let result = cwt(
      &signal,
      |points, scale| shannon(points, 1.0, 0.5, scale),
      &scales
  ).unwrap();
  ```

### Frequency B-Spline (FBSP) Wavelet

Combines a B-spline function with a complex modulation term, providing a controllable trade-off between time and frequency resolution through the order parameter.

- **Properties**:
  - Adjustable time-frequency resolution via order parameter
  - Higher orders produce smoother wavelets
  - Complex-valued
  - Good numerical stability
  
- **Use cases**:
  - Applications requiring customizable time-frequency trade-off
  - Spectral analysis with varying resolution requirements
  - Signal denoising
  - Feature extraction
  
- **Code example**:
  ```rust
  use scirs2_signal::wavelets::{cwt, fbsp};
  
  // CWT with FBSP wavelet, center_frequency=1.0, bandwidth=0.5, order=3
  let result = cwt(
      &signal,
      |points, scale| fbsp(points, 1.0, 0.5, 3, scale),
      &scales
  ).unwrap();
  ```

### Paul Wavelet

A complex wavelet with good time localization and poor frequency localization.

- **Properties**:
  - Complex-valued
  - Excellent time localization
  - Adjustable order parameter
  - Fast decay in frequency domain
  
- **Use cases**:
  - Detecting transient features
  - Time-localized events
  - Sharp peaks in signals
  
- **Code example**:
  ```rust
  use scirs2_signal::wavelets::{cwt, paul};
  
  // CWT with Paul wavelet of order 4
  let result = cwt(
      &signal,
      |points, scale| paul(points, 4, scale),
      &scales
  ).unwrap();
  ```

## Wavelet Selection Guide

Choosing the right wavelet depends on signal characteristics and application requirements:

1. **For sharp transitions or discontinuities**:
   - Haar wavelet
   - Lower-order Daubechies wavelets (DB2, DB3)
   - Paul wavelet (for CWT)

2. **For smooth signals**:
   - Higher-order Daubechies (DB8+)
   - Coiflets
   - Symlets
   - Complex Morlet (for CWT)

3. **For image processing**:
   - Biorthogonal wavelets (especially Bior3.5, Bior3.7)
   - Symlets

4. **For frequency analysis**:
   - Meyer or DMeyer wavelets
   - Shannon (for precise frequency in CWT)
   - FBSP (for tunable frequency resolution in CWT)

5. **For real-time processing (computational efficiency)**:
   - Haar wavelet (fastest)
   - Low-order Daubechies
   - DMeyer (over Meyer)

6. **For signal compression**:
   - Biorthogonal wavelets
   - Daubechies with appropriate vanishing moments

7. **For signals requiring phase information**:
   - Symlets
   - Biorthogonal wavelets
   - Complex wavelets (Complex Morlet, Complex Gaussian, etc.)

## Wavelet Transform Types

The scirs2-signal crate implements multiple types of wavelet transforms:

### Discrete Wavelet Transform (DWT)

The standard wavelet transform with downsampling at each level.

```rust
use scirs2_signal::dwt::{dwt_decompose, dwt_reconstruct, wavedec, waverec, Wavelet};

// Single-level decomposition
let (approx, detail) = dwt_decompose(&signal, Wavelet::DB(4), None).unwrap();

// Multi-level decomposition (3 levels)
let coeffs = wavedec(&signal, Wavelet::DB(4), Some(3), None).unwrap();
```

### Stationary Wavelet Transform (SWT)

Also known as the "à trous" algorithm or Undecimated Wavelet Transform. Provides translation invariance by eliminating downsampling.

```rust
use scirs2_signal::swt::{swt_decompose, swt_reconstruct, swt, iswt};
use scirs2_signal::dwt::Wavelet;

// Single-level SWT
let (approx, detail) = swt_decompose(&signal, Wavelet::DB(4), 1, None).unwrap();

// Multi-level SWT (3 levels)
let (details, approx) = swt(&signal, Wavelet::DB(4), 3, None).unwrap();
```

### Wavelet Packet Transform (WPT)

An extension of the DWT that decomposes both approximation and detail coefficients at each level, creating a full binary tree.

```rust
use scirs2_signal::wpt::{wp_decompose, reconstruct_from_nodes};
use scirs2_signal::dwt::Wavelet;

// Decompose to level 3
let tree = wp_decompose(&signal, Wavelet::DB(4), 3, None).unwrap();

// Reconstruct using specific nodes
let nodes = vec![(3, 0), (3, 1), (3, 2), (3, 3)];
let reconstructed = reconstruct_from_nodes(&tree, &nodes).unwrap();
```

## Performance Considerations

Performance can vary significantly between wavelet families:

1. **Filter length**: Wavelets with longer filters (higher order) require more computation.
   - Haar: 2 taps (fastest)
   - DB4: 8 taps
   - Meyer/DMeyer: 62 taps (slower)

2. **Transform type**:
   - DWT is the fastest (with downsampling)
   - SWT is slower (no downsampling, maintains original signal length)
   - WPT is slowest (full decomposition tree)

3. **Signal length**: Performance scales with signal length.

4. **Implementation efficiency**: The DMeyer wavelet is a more efficient alternative to the Meyer wavelet.

See [wavelet_benchmarks.md](wavelet_benchmarks.md) for detailed performance comparisons.

## Examples

### Basic Signal Denoising

```rust
use scirs2_signal::dwt::{wavedec, waverec, Wavelet};
use scirs2_signal::waveforms::{chirp, time_array};
use rand::Rng;

fn wavelet_denoising_example() {
    // Generate a chirp signal
    let fs = 1000.0;
    let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>();
    let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();
    
    // Add noise
    let mut rng = rand::rng();
    let noisy_signal = signal.iter()
        .map(|&x| x + 0.1 * rng.random_range(-1.0..1.0))
        .collect::<Vec<f64>>();
    
    // Decompose signal
    let coeffs = wavedec(&noisy_signal, Wavelet::Sym(4), Some(3), None).unwrap();
    
    // Apply thresholding to detail coefficients
    let mut denoised_coeffs = coeffs.clone();
    
    // Simple hard thresholding
    let threshold = 0.3;
    for i in 1..denoised_coeffs.len() {  // Skip approximation coeffs
        for val in denoised_coeffs[i].iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
            }
        }
    }
    
    // Reconstruct signal
    let denoised_signal = waverec(&denoised_coeffs, Wavelet::Sym(4)).unwrap();
}
```

### Feature Extraction

```rust
use scirs2_signal::dwt::{wavedec, Wavelet};
use scirs2_signal::waveforms::sine;

fn wavelet_feature_extraction() {
    // Generate two different signals
    let fs = 1000.0;
    let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>();
    
    let signal1 = sine(&t, 10.0, 0.0, 1.0).unwrap();  // 10 Hz
    let signal2 = sine(&t, 50.0, 0.0, 1.0).unwrap();  // 50 Hz
    
    // Extract wavelet features
    let coeffs1 = wavedec(&signal1, Wavelet::DB(4), Some(5), None).unwrap();
    let coeffs2 = wavedec(&signal2, Wavelet::DB(4), Some(5), None).unwrap();
    
    // Calculate energy in each subband
    let mut energy1 = Vec::new();
    let mut energy2 = Vec::new();
    
    for level in 0..coeffs1.len() {
        let e1: f64 = coeffs1[level].iter().map(|&x| x * x).sum();
        let e2: f64 = coeffs2[level].iter().map(|&x| x * x).sum();
        
        energy1.push(e1);
        energy2.push(e2);
    }
    
    // Energy distribution clearly differentiates between signals
    println!("Signal 1 (10 Hz) energy distribution: {:?}", energy1);
    println!("Signal 2 (50 Hz) energy distribution: {:?}", energy2);
}
```

### Wavelet-based Edge Detection

```rust
use scirs2_signal::dwt::{dwt_decompose, Wavelet};

fn edge_detection() {
    // Example signal with edges
    let signal = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 
        5.0, 5.0, 5.0, 5.0, 5.0,
        2.0, 2.0, 2.0, 2.0, 2.0,
        8.0, 8.0, 8.0, 8.0, 8.0,
    ];
    
    // Decompose using Haar (good for edge detection)
    let (_, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();
    
    // Find edges (high detail coefficients)
    let threshold = 2.0;
    let edges: Vec<usize> = detail.iter()
        .enumerate()
        .filter(|(_, &val)| val.abs() > threshold)
        .map(|(i, _)| i * 2)  // Adjust index due to downsampling
        .collect();
    
    println!("Edges detected at indices: {:?}", edges);
}
```

### Multi-resolution Analysis with SWT

```rust
use scirs2_signal::swt::{swt};
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::waveforms::chirp;

fn multiresolution_analysis() {
    // Generate a chirp signal with increasing frequency
    let fs = 1000.0;
    let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>();
    let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();
    
    // Perform 3-level SWT
    let (details, approx) = swt(&signal, Wavelet::DMeyer, 3, None).unwrap();
    
    // Analyze frequency content at different scales
    println!("Approximation coefficients represent low frequencies");
    println!("Detail level 3 (coarsest): very low frequency transitions");
    println!("Detail level 2: medium frequency transitions");
    println!("Detail level 1 (finest): high frequency transitions");
    
    // Calculate energy in each detail level
    for (i, detail) in details.iter().enumerate() {
        let energy: f64 = detail.iter().map(|&x| x * x).sum();
        println!("Energy in detail level {}: {}", i+1, energy);
    }
}
```