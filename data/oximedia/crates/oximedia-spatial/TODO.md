# oximedia-spatial TODO

## Current Status
- 8 source files (lib.rs + 7 modules): ambisonics, binaural, head_tracking, object_audio, room_simulation, vbap, wave_field
- Higher-Order Ambisonics (HOA) up to 3rd order encoding/decoding
- HRTF-based binaural rendering, image-source room acoustics, VBAP for 2D/3D loudspeaker arrays
- Quaternion-based head tracking, wave field synthesis, ADM object audio (Dolby Atmos / Auro-3D / DTS:X beds)
- Minimal dependencies: only thiserror

## Enhancements
- [x] Extend `AmbisonicsEncoder` to support 4th and 5th order HOA for large-venue immersive applications
- [x] Add near-field compensation filters to `ambisonics` for close-source rendering accuracy
- [x] Implement HRTF interpolation in `binaural` using spherical harmonics for smoother head rotation transitions
- [x] Add frequency-dependent wall absorption coefficients to `room_simulation` (currently likely uses single absorption value)
- [x] Extend `vbap` to support irregular speaker layouts with automatic triangulation (Delaunay-based)
- [x] Add IMU sensor fusion (accelerometer + gyroscope + magnetometer) to `head_tracking` complementary filter (verified 2026-05-16; src/imu_fusion.rs:15 Mahony filter, gyroscope/accelerometer/magnetometer fusion:50)
- [x] Implement distance attenuation models (inverse square, logarithmic, custom curve) in `object_audio` (verified 2026-05-16; src/distance_attenuation.rs:522 lines)
- [x] Add Doppler effect simulation for moving sound sources (verified 2026-05-16; src/doppler.rs:425 lines)

## New Features
- [x] Implement `dbap` (Distance-Based Amplitude Panning) module as alternative to VBAP for non-regular layouts (verified 2026-05-16; src/dbap.rs:326 lines)
- [x] Add `reverb` module with algorithmic late reverberation (Schroeder/Moorer style) separate from room simulation (verified 2026-05-16; src/reverb.rs:808 lines, Schroeder 1962 + Moorer 1979)
- [x] Implement `hoa_decoder` module with AllRAD (All-Round Ambisonic Decoding) for arbitrary speaker arrays (verified 2026-05-16; src/hoa_decoder.rs:516 lines)
- [x] Add `spatial_audio_format` module for ADM BWF (Broadcast Wave Format) and MPEG-H 3D Audio metadata I/O (verified 2026-05-16; src/adm_bwf.rs:653 lines)
- [x] Implement `acoustic_raytracer` module for geometric acoustics beyond image-source method (verified 2026-05-16; src/acoustic_raytracer.rs:543 lines)
- [x] Add `binauralizer` module that converts any multi-channel format to binaural using virtual speakers (verified 2026-05-16; src/binauralizer.rs:576 lines)
- [x] Implement `spatial_capture` module for A-format to B-format microphone array conversion (Soundfield, Eigenmike) (verified 2026-05-16; src/spatial_capture.rs:372 lines)
- [x] Add `zone_control` module for multi-zone spatial audio rendering in installations (verified 2026-05-16; src/zone_control.rs:573 lines)

## Performance
- [x] Use SIMD-accelerated spherical harmonic coefficient computation in `ambisonics` encoding (sh_dot_product added; AVX2+scalar; 2026-05-30)
- [x] Implement partitioned convolution in `binaural` HRTF rendering for lower latency (overlap-save with FFT) (partitioned_convolution.rs; oxifft overlap-save; 2026-05-30)
- [x] Cache VBAP gain matrices for static speaker layouts to avoid per-frame triangulation lookups (pre-computed at construction, tests added; 2026-05-30)
- [x] Use parallel ray casting in `room_simulation` for independent reflection path computation (rayon par_iter; 2026-05-30)
- [x] Pre-compute and cache head-tracking rotation matrices for common angle quantization steps (cached_rotation_matrix done)

## Testing
- [x] Add energy preservation tests for `AmbisonicsEncoder` (total energy in equals total energy out) (test_ambisonics_energy_preservation done)
- [ ] Test `binaural` rendering with known HRTF datasets (MIT KEMAR, CIPIC) and verify ITD/ILD values
- [x] Add convergence tests for `head_tracking` complementary filter with synthetic IMU data (verified: head_tracking.rs:1023:test_low_pass_filter_converges, :1099:test_head_tracker_accel_correction_at_low_alpha)
- [x] Test `vbap` panning law correctness for source positions at exact speaker locations (gain should be 1.0 at that speaker) (verified: vbap.rs:888:test_pan_exact_speaker_direction_full_gain, max_gain > 0.9)
- [x] Add round-trip test: encode mono source to Ambisonics then decode back, verify spectral similarity (test_ambisonics_encode_decode_roundtrip done)

## Documentation
- [ ] Add spatial coordinate convention diagram (azimuth, elevation, distance) used throughout the crate
- [ ] Document supported loudspeaker layout formats and how to define custom arrays for VBAP/WFS
- [ ] Add usage examples for integrating head_tracking with binaural rendering in real-time playback
