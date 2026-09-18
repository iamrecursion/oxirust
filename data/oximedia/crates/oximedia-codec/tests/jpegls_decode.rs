//! JPEG-LS decoder integration tests (ISO 14495-1).

#[cfg(feature = "jpegls")]
mod jpegls_tests {
    use oximedia_codec::jpegls::context::context_index;
    use oximedia_codec::jpegls::golomb::{
        decode_golomb_unsigned, map_error_lossless, unmap_error_lossless, BitReader,
    };
    use oximedia_codec::jpegls::predictor::predict;
    use oximedia_codec::jpegls::{JlsError, JpegLsDecoder, JpegLsEncoder, JpegLsEncoderConfig};

    // ─── lossless encoder helper for round-trip tests ────────────────────────
    //
    // Wave 6 + Wave 8 originally inlined a minimal §A.6-only encoder here.
    // Wave 10 Slice 3 promotes the decoder to §A.6 + §A.7 RUN mode; an
    // encoder that didn't emit §A.7 tokens would desync the round-trip.
    // The helper now delegates to the production `JpegLsEncoder`, which
    // is the canonical paired encoder.

    /// Encode a greyscale single-component lossless JPEG-LS stream by
    /// delegating to the production [`JpegLsEncoder`].
    fn encode_lossless_greyscale(
        samples: &[u16],
        width: u16,
        height: u16,
        precision: u8,
    ) -> Vec<u8> {
        let cfg = JpegLsEncoderConfig::greyscale(u32::from(width), u32::from(height), precision);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config valid");
        enc.encode_greyscale(samples).expect("encode greyscale ok")
    }

    // ─── test 1: detection ───────────────────────────────────────────────────

    #[test]
    fn is_jpegls_detection() {
        // A minimal header: SOI (FFD8) + SOF55 marker byte (FFF7)
        let valid = [
            0xFFu8, 0xD8, 0xFF, 0xF7, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11,
            0x00,
        ];
        assert!(JpegLsDecoder::is_jpegls(&valid));

        let not_valid = [0xFFu8, 0xD8, 0xFF, 0xC0]; // JPEG SOF0, not JLS
        assert!(!JpegLsDecoder::is_jpegls(&not_valid));

        let random = [0x00u8, 0x11, 0x22, 0x33, 0x44];
        assert!(!JpegLsDecoder::is_jpegls(&random));
    }

    // ─── test 2: rejection of non-JLS input ──────────────────────────────────

    #[test]
    fn not_jpegls_rejection() {
        // PNG header
        let png_header = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = JpegLsDecoder::decode(&png_header);
        assert!(matches!(result, Err(JlsError::NotJpegLs)));

        // Empty
        let empty: &[u8] = &[];
        let result = JpegLsDecoder::decode(empty);
        assert!(matches!(result, Err(JlsError::NotJpegLs)));

        // JPEG (not JPEG-LS) — starts with FFD8 FFE0 (JFIF APP0)
        let jpeg_header = [0xFFu8, 0xD8, 0xFF, 0xE0];
        let result = JpegLsDecoder::decode(&jpeg_header);
        // Should fail at header parsing (no SOF55 found before truncation)
        assert!(result.is_err());
    }

    // ─── test 3: round-trip encode/decode of constant 8×8 grey image ─────────

    #[test]
    fn round_trip_constant_grey_8x8() {
        let width = 8u16;
        let height = 8u16;
        let fill = 128u16;
        let samples: Vec<u16> = vec![fill; (width * height) as usize];

        let encoded = encode_lossless_greyscale(&samples, width, height, 8);
        let decoded =
            JpegLsDecoder::decode(&encoded).unwrap_or_else(|e| panic!("decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));
        assert_eq!(decoded.num_components, 1);
        assert_eq!(decoded.precision, 8);
        assert_eq!(decoded.samples.len(), 1);

        for (i, &px) in decoded.samples[0].iter().enumerate() {
            assert_eq!(px, fill, "pixel {i} mismatch: got {px}, expected {fill}");
        }
    }

    // ─── test 4: round-trip of a linear gradient (single row) ────────────────

    #[test]
    fn round_trip_linear_gradient_16x1() {
        let width = 16u16;
        let height = 1u16;
        let samples: Vec<u16> = (0u16..16).map(|i| i * 16).collect();

        let encoded = encode_lossless_greyscale(&samples, width, height, 8);
        let decoded =
            JpegLsDecoder::decode(&encoded).unwrap_or_else(|e| panic!("decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));

        for (i, (&original, &reconstructed)) in
            samples.iter().zip(decoded.samples[0].iter()).enumerate()
        {
            assert_eq!(
                reconstructed, original,
                "pixel {i}: expected {original}, got {reconstructed}"
            );
        }
    }

    // ─── test 5: predictor unit tests ─────────────────────────────────────────

    #[test]
    fn predictor_basic() {
        // c >= max(a,b): predict = min(a,b)
        assert_eq!(predict(10, 20, 25), 10);
        assert_eq!(predict(20, 10, 25), 10);

        // c <= min(a,b): predict = max(a,b)
        assert_eq!(predict(10, 20, 5), 20);
        assert_eq!(predict(20, 10, 5), 20);

        // c between a and b: predict = a + b - c
        assert_eq!(predict(10, 20, 15), 15);
        assert_eq!(predict(5, 15, 10), 10);

        // Edge: all equal
        assert_eq!(predict(7, 7, 7), 7);
    }

    // ─── test 6: context index sign normalisation ─────────────────────────────

    #[test]
    fn context_index_sign_normalisation() {
        // All-positive and all-negative triples must map to the same index.
        let (idx_pos, sign_pos) = context_index(1, 2, 3);
        let (idx_neg, sign_neg) = context_index(-1, -2, -3);
        assert_eq!(
            idx_pos, idx_neg,
            "positive and negative triples must share context"
        );
        assert_eq!(sign_pos, 1);
        assert_eq!(sign_neg, -1);

        // Mixed: first non-zero already positive.
        let (idx_a, sign_a) = context_index(1, -2, 3);
        let (idx_b, sign_b) = context_index(-1, 2, -3);
        assert_eq!(idx_a, idx_b);
        assert_eq!(sign_a, 1);
        assert_eq!(sign_b, -1);

        // All-zero: sign must be +1 (positive-normalised by convention).
        let (_, sign_zero) = context_index(0, 0, 0);
        assert_eq!(sign_zero, 1);
    }

    // ─── test 7: Golomb decode k=0 ────────────────────────────────────────────

    #[test]
    fn golomb_decode_k0_unary_three() {
        // k=0 is pure unary coding.
        // 3 zeros then a 1: bit pattern 0001xxxx → byte 0b00010000 = 0x10
        let data = [0b0001_0000u8];
        let mut reader = BitReader::new(&data);
        let val = decode_golomb_unsigned(&mut reader, 0);
        assert_eq!(val, Some(3));
    }

    #[test]
    fn golomb_decode_k0_unary_zero() {
        // The first bit is a 1: value = 0
        let data = [0b1000_0000u8];
        let mut reader = BitReader::new(&data);
        let val = decode_golomb_unsigned(&mut reader, 0);
        assert_eq!(val, Some(0));
    }

    // ─── test 8: Golomb decode k=2 ────────────────────────────────────────────

    #[test]
    fn golomb_decode_k2_value_five() {
        // value=5: unary = 5 >> 2 = 1, suffix = 5 & 3 = 1 (binary 01)
        // bits: 0 1 01 xxxx → 0b0101xxxx = 0x50
        let data = [0b0101_0000u8];
        let mut reader = BitReader::new(&data);
        let val = decode_golomb_unsigned(&mut reader, 2);
        assert_eq!(val, Some(5));
    }

    #[test]
    fn golomb_decode_k2_value_one() {
        // value=1: unary = 1>>2 = 0, suffix = 1&3 = 1 (binary 01)
        // bits: 1 01 xxxxx → 0b101_00000
        let data = [0b1010_0000u8];
        let mut reader = BitReader::new(&data);
        let val = decode_golomb_unsigned(&mut reader, 2);
        assert_eq!(val, Some(1));
    }

    // ─── test 9: full round-trip on non-trivial 8×8 image ────────────────────

    #[test]
    fn round_trip_encode_decode_8x8() {
        let width = 8u16;
        let height = 8u16;

        // Create a natural-image-like ramp with some variation.
        let mut samples = Vec::with_capacity(64);
        for row in 0u16..8 {
            for col in 0u16..8 {
                let val = ((row * 30 + col * 10 + 5) % 256) as u16;
                samples.push(val);
            }
        }

        let encoded = encode_lossless_greyscale(&samples, width, height, 8);
        let decoded =
            JpegLsDecoder::decode(&encoded).unwrap_or_else(|e| panic!("decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));
        assert_eq!(decoded.num_components, 1);

        for (i, (&original, &reconstructed)) in
            samples.iter().zip(decoded.samples[0].iter()).enumerate()
        {
            assert_eq!(
                reconstructed, original,
                "pixel {i}: encode→decode round-trip mismatch: expected {original}, got {reconstructed}"
            );
        }
    }

    // ─── test 10: 16-bit lossless round-trip ─────────────────────────────────

    #[test]
    fn round_trip_16bit_4x4() {
        let width = 4u16;
        let height = 4u16;
        // 12-bit samples (range 0..4095)
        let samples: Vec<u16> = (0u16..16).map(|i| i * 257).collect();

        let encoded = encode_lossless_greyscale(&samples, width, height, 12);
        let decoded =
            JpegLsDecoder::decode(&encoded).unwrap_or_else(|e| panic!("16-bit decode failed: {e}"));

        assert_eq!(decoded.precision, 12);
        for (i, (&orig, &rec)) in samples.iter().zip(decoded.samples[0].iter()).enumerate() {
            assert_eq!(rec, orig, "pixel {i}: 12-bit round-trip failed");
        }
    }

    // ─── test 11: unmap_error_lossless round-trip ─────────────────────────────

    #[test]
    fn unmap_error_roundtrip() {
        for err in -200i32..=200 {
            let mapped = map_error_lossless(err);
            let recovered = unmap_error_lossless(mapped);
            assert_eq!(recovered, err, "unmap(map({err})) = {recovered} != {err}");
        }
    }

    // ─── test 12: JpegLs CodecId properties ──────────────────────────────────

    #[test]
    fn codec_id_jpegls_properties() {
        use oximedia_core::types::{CodecId, MediaType};

        let codec = CodecId::JpegLs;
        assert_eq!(codec.media_type(), MediaType::Video);
        assert!(codec.is_video());
        assert!(!codec.is_audio());
        assert!(!codec.is_subtitle());
        assert!(codec.is_lossless());
        assert_eq!(codec.name(), "jpegls");
        assert_eq!(format!("{codec}"), "jpegls");
    }

    #[test]
    fn codec_id_jpegls_from_str() {
        use oximedia_core::types::CodecId;

        assert_eq!("jpegls".parse::<CodecId>().expect("parse"), CodecId::JpegLs);
        assert_eq!(
            "jpeg-ls".parse::<CodecId>().expect("parse"),
            CodecId::JpegLs
        );
        assert_eq!("jls".parse::<CodecId>().expect("parse"), CodecId::JpegLs);
        assert_eq!("JPEGLS".parse::<CodecId>().expect("parse"), CodecId::JpegLs);
        assert_eq!("JLS".parse::<CodecId>().expect("parse"), CodecId::JpegLs);
    }

    #[test]
    fn codec_matrix_jpegls_containers() {
        use oximedia_core::{codec_matrix::CodecMatrix, types::CodecId};

        assert!(CodecMatrix::is_compatible(CodecId::JpegLs, "jls"));
        assert!(CodecMatrix::is_compatible(CodecId::JpegLs, "dicom"));
        assert!(CodecMatrix::is_compatible(CodecId::JpegLs, "dcm"));
        assert!(CodecMatrix::is_compatible(CodecId::JpegLs, "jpg"));
        assert!(!CodecMatrix::is_compatible(CodecId::JpegLs, "mp4"));
        assert!(!CodecMatrix::is_compatible(CodecId::JpegLs, "webm"));

        let containers = CodecMatrix::compatible_containers(CodecId::JpegLs);
        assert!(!containers.is_empty());
        assert!(containers.contains(&"jls"));
        assert!(containers.contains(&"dicom"));
    }

    // ─── near-lossless encoder helper ────────────────────────────────────────
    //
    // Wave 6 + Wave 8 originally inlined a minimal §A.6-only near-lossless
    // encoder here.  Wave 10 promotes the decoder to §A.6 + §A.7 RUN mode,
    // so the encoder must also emit §A.7 tokens — the production
    // `JpegLsEncoder` is the canonical paired encoder and is delegated to
    // here.

    /// Encode a single-component near-lossless JPEG-LS stream by
    /// delegating to the production [`JpegLsEncoder`].
    fn encode_near_lossless_greyscale(
        samples: &[u16],
        width: u16,
        height: u16,
        precision: u8,
        near: u8,
    ) -> Vec<u8> {
        let cfg = JpegLsEncoderConfig::greyscale(u32::from(width), u32::from(height), precision)
            .with_near(near);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config valid");
        enc.encode_greyscale(samples)
            .expect("encode near-lossless greyscale ok")
    }

    // ─── interleaved multi-component encoder helper ───────────────────────────

    /// Encode a multi-component lossless JPEG-LS stream (ILV = 0/1/2) by
    /// delegating to the production [`JpegLsEncoder`].
    fn encode_lossless_multicomponent(
        samples: &[Vec<u16>],
        width: u16,
        height: u16,
        precision: u8,
        ilv: u8,
    ) -> Vec<u8> {
        let nc = samples.len() as u8;
        let cfg = JpegLsEncoderConfig::multicomponent(
            u32::from(width),
            u32::from(height),
            nc,
            precision,
            ilv,
        );
        let enc = JpegLsEncoder::new(cfg).expect("encoder config valid");
        let plane_refs: Vec<&[u16]> = samples.iter().map(Vec::as_slice).collect();
        enc.encode_planes(&plane_refs)
            .expect("encode multicomponent ok")
    }

    // ─── test 13: near-lossless NEAR=1 round-trip (8×8 greyscale) ────────────

    #[test]
    fn round_trip_near_lossless_1_grey_8x8() {
        let width = 8u16;
        let height = 8u16;
        let near: u8 = 1;

        // Natural-image-like source with values spanning the full 8-bit range.
        let mut samples = Vec::with_capacity(64);
        for row in 0u16..8 {
            for col in 0u16..8 {
                let val = ((row * 37 + col * 19 + 13) % 256) as u16;
                samples.push(val);
            }
        }

        let encoded = encode_near_lossless_greyscale(&samples, width, height, 8, near);
        let decoded = JpegLsDecoder::decode(&encoded)
            .unwrap_or_else(|e| panic!("near-lossless NEAR=1 decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));
        assert_eq!(decoded.num_components, 1);
        assert_eq!(decoded.precision, 8);
        assert_eq!(decoded.samples.len(), 1);

        for (i, (&original, &reconstructed)) in
            samples.iter().zip(decoded.samples[0].iter()).enumerate()
        {
            let diff = (original as i32 - reconstructed as i32).abs();
            assert!(
                diff <= near as i32,
                "pixel {i}: |{original} - {reconstructed}| = {diff} > NEAR={near}"
            );
        }
    }

    // ─── test 14: near-lossless NEAR=2 round-trip (16×1 gradient) ────────────

    #[test]
    fn round_trip_near_lossless_2_gradient_16x1() {
        let width = 16u16;
        let height = 1u16;
        let near: u8 = 2;

        // A linear gradient from 0 to 240 in steps of 16.
        let samples: Vec<u16> = (0u16..16).map(|i| i * 16).collect();

        let encoded = encode_near_lossless_greyscale(&samples, width, height, 8, near);
        let decoded = JpegLsDecoder::decode(&encoded)
            .unwrap_or_else(|e| panic!("near-lossless NEAR=2 decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));
        assert_eq!(decoded.num_components, 1);

        for (i, (&original, &reconstructed)) in
            samples.iter().zip(decoded.samples[0].iter()).enumerate()
        {
            let diff = (original as i32 - reconstructed as i32).abs();
            assert!(
                diff <= near as i32,
                "pixel {i}: |{original} - {reconstructed}| = {diff} > NEAR={near}"
            );
        }
    }

    // ─── test 15: ILV=1 (line-interleaved) RGB 4×4 lossless round-trip ────────

    #[test]
    fn round_trip_ilv1_rgb_4x4() {
        let width = 4u16;
        let height = 4u16;

        // Three independent component planes with distinct patterns.
        let comp_r: Vec<u16> = (0u16..16).map(|i| (i * 17) % 256).collect();
        let comp_g: Vec<u16> = (0u16..16).map(|i| (i * 31 + 50) % 256).collect();
        let comp_b: Vec<u16> = (0u16..16).map(|i| (i * 7 + 120) % 256).collect();
        let planes = vec![comp_r.clone(), comp_g.clone(), comp_b.clone()];

        let encoded = encode_lossless_multicomponent(&planes, width, height, 8, 1);
        let decoded = JpegLsDecoder::decode(&encoded)
            .unwrap_or_else(|e| panic!("ILV=1 RGB decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.samples.len(), 3);

        for (ci, (original_plane, decoded_plane)) in
            planes.iter().zip(decoded.samples.iter()).enumerate()
        {
            for (pi, (&orig, &rec)) in original_plane.iter().zip(decoded_plane.iter()).enumerate() {
                assert_eq!(
                    rec, orig,
                    "ILV=1 component {ci} pixel {pi}: expected {orig}, got {rec}"
                );
            }
        }
    }

    // ─── test 16: ILV=2 (sample-interleaved) RGB 2×2 lossless round-trip ──────

    #[test]
    fn round_trip_ilv2_rgb_2x2() {
        let width = 2u16;
        let height = 2u16;

        // Four pixels, three components each.
        let comp_r: Vec<u16> = vec![100, 150, 200, 250];
        let comp_g: Vec<u16> = vec![10, 60, 110, 160];
        let comp_b: Vec<u16> = vec![55, 85, 115, 145];
        let planes = vec![comp_r.clone(), comp_g.clone(), comp_b.clone()];

        let encoded = encode_lossless_multicomponent(&planes, width, height, 8, 2);
        let decoded = JpegLsDecoder::decode(&encoded)
            .unwrap_or_else(|e| panic!("ILV=2 RGB decode failed: {e}"));

        assert_eq!(decoded.width, u32::from(width));
        assert_eq!(decoded.height, u32::from(height));
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.samples.len(), 3);

        for (ci, (original_plane, decoded_plane)) in
            planes.iter().zip(decoded.samples.iter()).enumerate()
        {
            for (pi, (&orig, &rec)) in original_plane.iter().zip(decoded_plane.iter()).enumerate() {
                assert_eq!(
                    rec, orig,
                    "ILV=2 component {ci} pixel {pi}: expected {orig}, got {rec}"
                );
            }
        }
    }
}
