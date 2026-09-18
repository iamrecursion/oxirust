# Signal Transforms Module - v0.2.0

**Production-ready signal transformation capabilities for scirs2-transform**

## Overview

The `signal_transforms` module provides comprehensive signal processing transforms for time-frequency analysis, wavelet analysis, and audio feature extraction. This module is designed to match and exceed the capabilities of SciPy's signal processing and librosa audio analysis libraries.

## Features

### 1. Discrete Wavelet Transform (DWT)

**Module**: `signal_transforms::dwt`

Provides 1D, 2D, and N-D discrete wavelet transforms with multiple wavelet families:

- **Wavelet Families**:
  - Haar (Daubechies-1)
  - Daubechies (N = 2, 4, 6, 8, 10, ...)
  - Symlet
  - Coiflet
  - Biorthogonal

- **Features**:
  - Single-level decomposition and reconstruction
  - Multi-level decomposition (wavedec/waverec)
  - 2D image decomposition (LL, LH, HL, HH coefficients)
  - Multiple boundary modes (zero, constant, symmetric, periodic, reflect)
  - Efficient SIMD-optimized convolution

- **Example**:
```rust
use scirs2_transform::{DWT, WaveletType};

let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let dwt = DWT::new(WaveletType::Haar)?.with_level(2);

// Multi-level decomposition
let coeffs = dwt.wavedec(&signal.view())?;

// Reconstruction
let reconstructed = dwt.waverec(&coeffs)?;
```

### 2. Continuous Wavelet Transform (CWT)

**Module**: `signal_transforms::cwt`

Time-frequency analysis using continuous wavelets:

- **Mother Wavelets**:
  - Morlet wavelet
  - Mexican Hat (Ricker) wavelet
  - Complex Morlet wavelet
  - Gaussian derivatives

- **Features**:
  - Direct convolution method
  - FFT-based fast CWT
  - Scalogram computation
  - Logarithmic scale spacing
  - Custom wavelet implementation support

- **Example**:
```rust
use scirs2_transform::{CWT, MorletWavelet};

let signal = Array1::from_vec((0..512).map(|i| (i as f64 * 0.1).sin()).collect());
let wavelet = MorletWavelet::default();
let cwt = CWT::with_log_scales(wavelet, 32, 1.0, 32.0);

// Compute scalogram
let scalogram = cwt.scalogram(&signal.view())?;
```

### 3. Wavelet Packet Transform (WPT)

**Module**: `signal_transforms::wpt`

Full wavelet packet decomposition with best basis selection:

- **Features**:
  - Full binary tree decomposition
  - Best basis selection using Shannon entropy
  - Multiple cost functions (Shannon, Threshold, LogEnergy, SURE)
  - Adaptive tree pruning
  - Denoising applications

- **Example**:
```rust
use scirs2_transform::{WPT, WaveletType};

let signal = Array1::from_vec((0..256).map(|i| (i as f64 * 0.05).sin()).collect());
let mut wpt = WPT::new(WaveletType::Daubechies(4), 4);

// Full decomposition
wpt.decompose(&signal.view())?;

// Select best basis
let best_basis = wpt.best_basis()?;
```

### 4. Short-Time Fourier Transform (STFT) and Spectrograms

**Module**: `signal_transforms::stft`

Time-frequency analysis using windowed Fourier transforms:

- **Window Functions**:
  - Hann
  - Hamming
  - Blackman
  - Bartlett
  - Kaiser
  - Tukey
  - Rectangular

- **Features**:
  - Forward and inverse STFT
  - Multiple padding modes
  - Configurable window size and hop size
  - Zero-padding support
  - Perfect reconstruction (with proper parameters)

- **Spectrogram Types**:
  - Power spectrum (magnitude squared)
  - Magnitude spectrum
  - Decibel scale (10 * log10)

- **Example**:
```rust
use scirs2_transform::{STFT, Spectrogram, SpectrogramScaling};

// STFT
let signal = Array1::from_vec((0..1024).map(|i| (i as f64 * 0.1).sin()).collect());
let stft = STFT::with_params(256, 128);

let coeffs = stft.transform(&signal.view())?;
let reconstructed = stft.inverse(&coeffs)?;

// Spectrogram
let config = STFTConfig {
    window_size: 512,
    hop_size: 256,
    ..Default::default()
};

let spec = Spectrogram::new(config)
    .with_scaling(SpectrogramScaling::Decibel)
    .compute(&signal.view())?;
```

### 5. Mel-Frequency Cepstral Coefficients (MFCC)

**Module**: `signal_transforms::mfcc`

Audio feature extraction for speech and music processing:

- **Features**:
  - Mel-scale filterbank
  - DCT-II transformation
  - Liftering (cepstral filtering)
  - Mean normalization
  - Delta and delta-delta features
  - Customizable configuration

- **Default Configuration**:
  - 13 MFCCs
  - 40 mel filters
  - 512-point FFT
  - 16 kHz sampling rate
  - 0-8000 Hz frequency range

- **Example**:
```rust
use scirs2_transform::{MFCC, MFCCConfig};

let signal = Array1::from_vec((0..16000).map(|i| (i as f64 * 0.01).sin()).collect());

// Extract MFCCs
let mfcc = MFCC::default()?;
let features = mfcc.extract(&signal.view())?; // 13 x N_frames

// Extract with deltas
let features_full = mfcc.extract_with_deltas(&signal.view())?; // 39 x N_frames

// Custom configuration
let config = MFCCConfig {
    n_mfcc: 20,
    n_mels: 80,
    sample_rate: 44100.0,
    ..Default::default()
};

let custom_mfcc = MFCC::new(config)?;
```

### 6. Constant-Q Transform (CQT) and Chromagram

**Module**: `signal_transforms::cqt`

Musically-motivated time-frequency analysis:

- **CQT Features**:
  - Logarithmically-spaced frequency bins
  - Constant Q-factor across frequencies
  - Customizable frequency range
  - Bins per octave configuration
  - Multiple window functions

- **Chromagram Features**:
  - 12-bin pitch class profiles
  - L1 and L2 normalization
  - Octave folding
  - Standard note labels (C, C#, D, ...)

- **Example**:
```rust
use scirs2_transform::{CQT, Chromagram, CQTConfig};

let signal = Array1::from_vec((0..22050).map(|i| {
    (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 22050.0).sin()
}).collect());

// CQT
let config = CQTConfig {
    sample_rate: 22050.0,
    fmin: 55.0, // A1
    bins_per_octave: 12,
    n_octaves: 7,
    ..Default::default()
};

let cqt = CQT::new(config)?;
let magnitude = cqt.magnitude(&signal.view())?;

// Chromagram
let chroma = Chromagram::default()?;
let chroma_features = chroma.compute(&signal.view())?; // 12 x N_frames

// L2-normalized chromagram
let chroma_norm = chroma.compute_normalized(&signal.view())?;
```

## Integration with scirs2-fft

All FFT-based operations use OxiFFT (COOLJAPAN Pure Rust FFT library) through scirs2-fft:

- CWT FFT-based computation
- STFT forward and inverse transforms
- MFCC power spectrum computation
- Implicit in mel filterbank application

## SIMD Optimization

Where applicable, the signal transforms leverage SIMD operations through scirs2-core:

- Convolution operations in DWT
- Element-wise operations in window functions
- Power spectrum computation
- Filterbank application

## Performance Characteristics

### DWT
- **Time Complexity**: O(N log N) for multilevel decomposition
- **Space Complexity**: O(N)
- **Optimal For**: Signals with 2^n length

### CWT
- **Time Complexity**: O(S * N^2) direct, O(S * N log N) FFT-based
- **Space Complexity**: O(S * N) where S is number of scales
- **Optimal For**: FFT method for signals > 256 samples

### WPT
- **Time Complexity**: O(N log N * L) where L is decomposition level
- **Space Complexity**: O(N * 2^L)
- **Optimal For**: Best basis selection with L <= 5

### STFT
- **Time Complexity**: O(F * M log M) where F is number of frames
- **Space Complexity**: O(F * M)
- **Optimal For**: Window size 256-2048, hop size = window_size/2

### MFCC
- **Time Complexity**: O(F * M log M + F * K) where K is mel filters
- **Space Complexity**: O(F * C) where C is number of MFCCs
- **Optimal For**: Audio signals at 16-48 kHz

### CQT
- **Time Complexity**: O(B * N) where B is number of bins
- **Space Complexity**: O(B * F) where F is number of frames
- **Optimal For**: Music analysis, harmonic content

## SciPy/librosa Equivalents

| scirs2-transform | SciPy/librosa | Description |
|------------------|---------------|-------------|
| `DWT::wavedec` | `pywt.wavedec` | Multilevel DWT |
| `DWT2D::wavedec2` | `pywt.wavedec2` | 2D multilevel DWT |
| `CWT::transform_fft` | `scipy.signal.cwt` | Continuous wavelet transform |
| `WPT::best_basis` | `pywt.WaveletPacket.get_level` | Best basis selection |
| `STFT::transform` | `scipy.signal.stft` | Short-time Fourier transform |
| `Spectrogram::compute` | `scipy.signal.spectrogram` | Spectrogram |
| `MFCC::extract` | `librosa.feature.mfcc` | MFCC extraction |
| `MFCC::delta` | `librosa.feature.delta` | Delta features |
| `CQT::transform` | `librosa.cqt` | Constant-Q transform |
| `Chromagram::compute` | `librosa.feature.chroma_cqt` | Chromagram |

## Testing

Comprehensive test coverage includes:

- Unit tests for each transform
- Integration tests for complete workflows
- Property-based tests for reconstruction accuracy
- Performance benchmarks vs SciPy

Run tests:
```bash
cargo test -p scirs2-transform signal_transforms
cargo test -p scirs2-transform --test signal_transforms_integration
```

Run benchmarks:
```bash
cargo bench -p scirs2-transform --bench signal_transforms_scipy_bench
```

## Error Handling

All transforms use `Result<T, TransformError>` for proper error handling:

```rust
use scirs2_transform::error::{Result, TransformError};

match dwt.wavedec(&signal.view()) {
    Ok(coeffs) => println!("Decomposition successful"),
    Err(TransformError::InvalidInput(msg)) => eprintln!("Invalid input: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Best Practices

### DWT
- Use power-of-2 signal lengths when possible
- Choose wavelet family based on signal characteristics:
  - Haar: Fast, discontinuous signals
  - Daubechies: General purpose, smooth signals
  - Symlets: Nearly symmetric, better phase characteristics
  - Coiflets: Balanced approximation and detail

### CWT
- Use FFT method for signals longer than 256 samples
- Choose scale spacing based on frequency resolution needs
- Morlet wavelet: Good time-frequency localization
- Mexican Hat: Good for edge detection

### STFT
- Window size: Trade-off between time and frequency resolution
- Hop size: Use 50% overlap (hop_size = window_size/2) for good reconstruction
- Hann/Hamming windows: General purpose
- Blackman window: Better frequency resolution

### MFCC
- 13 MFCCs standard for speech recognition
- Include delta and delta-delta for dynamic features
- Normalize features for machine learning applications
- Adjust frequency range based on signal content

### CQT/Chromagram
- Use for musical analysis and pitch tracking
- 12 bins per octave standard for Western music
- Higher bins/octave (24, 36) for more detailed analysis
- Normalize chromagrams for chord recognition

## Future Enhancements

- [ ] GPU acceleration for large-scale transforms
- [ ] Distributed processing for batch operations
- [ ] Real-time streaming support
- [ ] Additional wavelet families (Meyer, Morlet-like)
- [ ] Synchrosqueezing transforms
- [ ] Empirical mode decomposition (EMD)
- [ ] Hilbert-Huang transform
- [ ] Variable-Q transform (VQT)

## References

1. Mallat, S. (2008). A Wavelet Tour of Signal Processing (3rd ed.)
2. Daubechies, I. (1992). Ten Lectures on Wavelets
3. Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal Processing
4. Müller, M. (2015). Fundamentals of Music Processing
5. SciPy Signal Processing Documentation
6. librosa Audio Analysis Documentation

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)

---

**v0.2.0** - Production-ready signal transforms for scientific computing and audio analysis
