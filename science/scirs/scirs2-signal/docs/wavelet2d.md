# 2D Wavelet Transforms in scirs2-signal

This document provides an overview of the 2D wavelet transform implementation in scirs2-signal, including its applications, usage examples, and performance considerations.

## Introduction to 2D Wavelet Transforms

The 2D wavelet transform extends the concepts of 1D wavelet analysis to two-dimensional data, such as images. It provides a multi-resolution representation that captures both spatial and frequency information. The 2D DWT decomposes an image into four subbands:

- **LL (Approximation)**: Low-frequency components in both horizontal and vertical directions
- **LH (Horizontal Detail)**: Low-frequency in horizontal, high-frequency in vertical direction
- **HL (Vertical Detail)**: High-frequency in horizontal, low-frequency in vertical direction
- **HH (Diagonal Detail)**: High-frequency components in both directions

For multi-level decomposition, the LL subband is recursively decomposed into further subbands, creating a pyramid-like structure.

## Applications

2D wavelet transforms are widely used in:

1. **Image Compression**: Wavelets form the basis of modern image compression standards like JPEG2000
2. **Image Denoising**: Using wavelet thresholding techniques to remove noise while preserving edges
3. **Feature Extraction**: For texture analysis and pattern recognition
4. **Edge Detection**: Detail coefficients highlight edges and features at different scales
5. **Medical Imaging**: For analysis and enhancement of medical images
6. **Computer Vision**: For multi-scale feature detection and image segmentation

## Available Functions

The `dwt2d` module provides the following main functions:

- `dwt2d_decompose`: Single-level 2D wavelet decomposition
- `dwt2d_reconstruct`: Single-level 2D wavelet reconstruction
- `wavedec2`: Multi-level 2D wavelet decomposition
- `waverec2`: Multi-level 2D wavelet reconstruction

## Basic Usage

### Single-Level Decomposition and Reconstruction

```rust
use ndarray::Array2;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};

// Create or load an image
let image = Array2::from_shape_vec((4, 4), vec![
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0
]).unwrap();

// Decompose using Haar wavelet
let decomposition = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();

// Access the subbands
println!("Approximation coefficients (LL):");
println!("{:?}", decomposition.approx);

println!("Horizontal detail coefficients (LH):");
println!("{:?}", decomposition.detail_h);

// Reconstruct the image
let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();
```

### Multi-Level Decomposition and Reconstruction

```rust
use ndarray::Array2;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::dwt2d::{wavedec2, waverec2};

// Create or load an image
let image = /* ... */;

// Perform 3-level decomposition using DB4 wavelet
let coeffs = wavedec2(&image, Wavelet::DB(4), 3, None).unwrap();

// Access the coefficients at different levels
println!("Deepest level approximation (smallest scale):");
println!("{:?}", coeffs[0].approx);

// Reconstruct the image
let reconstructed = waverec2(&coeffs, Wavelet::DB(4), None).unwrap();
```

## Image Compression Example

A common application of 2D wavelets is image compression. Here's a simple example:

```rust
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::dwt2d::{wavedec2, waverec2, Dwt2dResult};

// Decompose image
let coeffs = wavedec2(&image, Wavelet::DB(4), 3, None).unwrap();

// Apply thresholding to detail coefficients
let mut thresholded_coeffs = coeffs.clone();
let threshold = 0.1;

for level in &mut thresholded_coeffs {
    // Zero out small detail coefficients
    for val in level.detail_h.iter_mut() {
        if val.abs() < threshold {
            *val = 0.0;
        }
    }
    // Same for detail_v and detail_d
}

// Reconstruct from thresholded coefficients
let compressed = waverec2(&thresholded_coeffs, Wavelet::DB(4), None).unwrap();
```

## Wavelet Selection for Images

Different wavelet families offer various trade-offs for image processing:

1. **Haar**: Simplest wavelet; good for edges but may introduce blocking artifacts
2. **Daubechies (DB4, DB6)**: Popular for image compression due to good localization properties
3. **Symlets**: Similar to Daubechies but more symmetrical, reducing phase distortion
4. **Biorthogonal**: Excellent for image compression due to symmetry and compact support
5. **Coiflets**: Designed for both good approximation and compression properties

## Performance Considerations

1. **Memory Usage**: The 2D DWT requires additional memory for storing the four subbands.
2. **Computational Complexity**: On the order of O(N) where N is the number of pixels in the image.
3. **Boundary Handling**: The `mode` parameter controls how image boundaries are handled. Options include "symmetric", "periodic", etc.
4. **Filter Length**: Longer filters (higher-order wavelets) require more computation but may provide better compression or analysis properties.

## Future Enhancements

Planned enhancements to the 2D wavelet transform implementation include:

1. **Parallel Processing**: Using rayon for parallel computation of rows and columns
2. **SIMD Optimization**: For faster convolution operations
3. **In-place Transforms**: Reducing memory requirements
4. **Specialized Image Compression Functions**: Including quantization and encoding
5. **Support for Color Images**: Processing multi-channel images
6. **Integration with scirs2-ndimage**: For seamless image processing workflows