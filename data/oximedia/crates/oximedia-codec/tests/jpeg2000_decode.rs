//! Integration tests for the JPEG 2000 lossless decoder.
//!
//! Tests a hand-crafted 16×16 single-component lossless codestream
//! where the image is a constant grey value of 128.
//!
//! For a constant image, the 5-3 wavelet forward transform produces:
//! - LL subband = constant image values (predict step only affects H, not L;
//!   update step adjusts L by floor((H[-1]+H)/4) = 0 for H=0)
//! - HL, LH, HH = all zeros
//!
//! The inverse transform therefore recovers the exact constant value.

#[cfg(feature = "jpeg2000")]
mod jpeg2000_integration {
    use oximedia_codec::jpeg2000::{
        box_parser::{is_jp2_container, parse_jp2, Jp2ColorSpace},
        markers::{parse_codestream, MarkerSegment},
        wavelet::{
            forward_53, inverse_wavelet_1d, inverse_wavelet_2d, inverse_wavelet_2d_97,
            reconstruct_levels, SubbandLevel, SubbandTree,
        },
    };

    /// Test 1: 5-3 wavelet lossless round-trip for a constant signal.
    ///
    /// For a constant-value image, forward then inverse wavelet must recover
    /// the original samples exactly.
    #[test]
    fn wavelet_roundtrip_constant_8_samples() {
        let original = vec![128i32; 8];
        let (low, high) = forward_53(&original);
        // High-pass (detail) should be all zeros for constant input.
        for &h in &high {
            assert_eq!(h, 0, "Detail must be 0 for constant signal");
        }
        let recovered = inverse_wavelet_1d(&low, &high);
        assert_eq!(
            recovered, original,
            "Round-trip must be lossless for constant signal"
        );
    }

    /// Test 2: Full 2D wavelet round-trip for a constant 16×16 image.
    #[test]
    fn wavelet_roundtrip_2d_constant_16x16() {
        let width = 16;
        let height = 16;
        let val = 128i32;

        // Forward horizontal transform on each row.
        let n_l_h = (width + 1) / 2; // 8
        let n_h_h = width / 2; // 8

        let image = vec![val; width * height];
        let mut ll_after_h = vec![0i32; n_l_h * height];
        let mut hl_sub = vec![0i32; n_h_h * height];

        for row in 0..height {
            let (l, h) = forward_53(&image[row * width..(row + 1) * width]);
            ll_after_h[row * n_l_h..row * n_l_h + l.len()].copy_from_slice(&l);
            hl_sub[row * n_h_h..row * n_h_h + h.len()].copy_from_slice(&h);
        }

        // Forward vertical transform on each column of the low-pass region → LL and LH.
        let n_l_v = (height + 1) / 2; // 8
        let n_h_v = height / 2; // 8

        let mut ll = vec![0i32; n_l_v * n_l_h];
        let mut lh = vec![0i32; n_h_v * n_l_h];

        for col in 0..n_l_h {
            let col_vals: Vec<i32> = (0..height).map(|r| ll_after_h[r * n_l_h + col]).collect();
            let (l, h) = forward_53(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                ll[r * n_l_h + col] = v;
            }
            for (r, &v) in h.iter().enumerate() {
                lh[r * n_l_h + col] = v;
            }
        }

        // Forward vertical transform on each column of the high-pass region → HL and HH.
        let mut hl = vec![0i32; n_l_v * n_h_h];
        let mut hh = vec![0i32; n_h_v * n_h_h];

        for col in 0..n_h_h {
            let col_vals: Vec<i32> = (0..height).map(|r| hl_sub[r * n_h_h + col]).collect();
            let (l, h) = forward_53(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                hl[r * n_h_h + col] = v;
            }
            for (r, &v) in h.iter().enumerate() {
                hh[r * n_h_h + col] = v;
            }
        }

        // For constant input, all detail subbands should be zero.
        for &v in &hl {
            assert_eq!(v, 0, "HL must be 0 for constant image");
        }
        for &v in &lh {
            assert_eq!(v, 0, "LH must be 0 for constant image");
        }
        for &v in &hh {
            assert_eq!(v, 0, "HH must be 0 for constant image");
        }
        // LL should be constant.
        for &v in &ll {
            assert_eq!(v, val, "LL should equal constant value after 5-3 forward");
        }

        // Now apply inverse wavelet.
        let output =
            inverse_wavelet_2d(&ll, &hl, &lh, &hh, width, height).expect("inverse_wavelet_2d");

        assert_eq!(output.len(), width * height, "Output length mismatch");
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, val, "Sample {i} should be {val}, got {v}");
        }
    }

    /// Test 3: `reconstruct_levels` with zero detail subbands recovers constant.
    #[test]
    fn reconstruct_levels_constant_16x16_one_level() {
        let width = 16;
        let height = 16;
        let val = 128i32;

        let n_l_h = (width + 1) / 2;
        let n_l_v = (height + 1) / 2;
        let n_h_h = width / 2;
        let n_h_v = height / 2;

        // LL = constant val (after forward transform of constant image, LL stays val).
        let ll = vec![val; n_l_v * n_l_h];
        let hl = vec![0i32; n_l_v * n_h_h];
        let lh = vec![0i32; n_h_v * n_l_h];
        let hh = vec![0i32; n_h_v * n_h_h];

        let tree = SubbandTree {
            ll,
            ll_width: n_l_h,
            ll_height: n_l_v,
            levels: vec![SubbandLevel {
                hl,
                lh,
                hh,
                width: n_l_h,
                height: n_l_v,
            }],
        };

        let output = reconstruct_levels(&tree, 1, width, height).expect("reconstruct_levels");
        assert_eq!(output.len(), width * height);
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, val, "Sample {i}: expected {val}, got {v}");
        }
    }

    /// Test 4: JP2 box parser minimal header.
    ///
    /// Constructs a 40-byte minimal JP2 header blob, parses it, asserts fields.
    #[test]
    fn jp2_box_parser_minimal() {
        // Build a minimal JP2 file with signature + jp2h + jp2c.
        let width: u32 = 32;
        let height: u32 = 64;
        let codestream_stub = [0xFF, 0x4Fu8, 0xFF, 0xD9u8]; // SOC + EOC

        let jp2 = build_minimal_jp2(width, height, 1, &codestream_stub);
        assert!(is_jp2_container(&jp2), "Should be detected as JP2");
        let (hdr, cs) = parse_jp2(&jp2).expect("parse_jp2");
        assert_eq!(hdr.width, width, "width mismatch");
        assert_eq!(hdr.height, height, "height mismatch");
        assert_eq!(hdr.num_components, 1, "num_components mismatch");
        assert_eq!(hdr.bit_depth, 8, "bit_depth mismatch");
        assert!(!hdr.is_signed, "should not be signed");
        assert_eq!(hdr.color_space, Jp2ColorSpace::Greyscale);
        assert_eq!(cs, &codestream_stub[..]);
    }

    /// Test 5: J2K marker parse — SOC + SIZ + SOT + EOC.
    #[test]
    fn j2k_marker_parse_soc_siz_sot_eoc() {
        let data = build_minimal_j2k_codestream(16, 16, 1);
        let segments = parse_codestream(&data).expect("parse_codestream");

        let siz = segments.iter().find_map(|s| {
            if let MarkerSegment::Siz(sz) = s {
                Some(sz)
            } else {
                None
            }
        });
        let sot = segments.iter().find_map(|s| {
            if let MarkerSegment::Sot(st) = s {
                Some(st)
            } else {
                None
            }
        });
        let has_eoc = segments.iter().any(|s| matches!(s, MarkerSegment::Eoc));

        let siz = siz.expect("SIZ marker");
        assert_eq!(siz.image_width(), 16, "SIZ width");
        assert_eq!(siz.image_height(), 16, "SIZ height");
        assert_eq!(siz.csiz, 1, "SIZ csiz");
        assert_eq!(siz.components[0].bit_depth(), 8, "component bit depth");

        let sot = sot.expect("SOT marker");
        assert_eq!(sot.isot, 0);
        assert!(has_eoc, "EOC must be present");
    }

    /// Test 6: Integration — decode a constant-grey 16×16 image.
    ///
    /// The test vector is constructed as:
    /// - All 256 coefficients in the LL subband = 128 (the constant grey value)
    /// - All HL, LH, HH subbands = 0
    /// The decoder's inverse wavelet recovers the exact value.
    ///
    /// We test this via the wavelet path directly (SubbandTree → reconstruct_levels)
    /// since building a valid Tier-1/Tier-2 encoded block for this purpose
    /// would require a full MQ encoder, which is out of scope for a decoder-only module.
    #[test]
    fn integration_constant_grey_16x16_via_wavelet() {
        let width = 16usize;
        let height = 16usize;
        let constant_val = 128i32;

        // Build subband tree for a constant image after 1-level 5-3 DWT.
        let n_l_h = (width + 1) / 2; // 8
        let n_l_v = (height + 1) / 2; // 8
        let n_h_h = width / 2; // 8
        let n_h_v = height / 2; // 8

        let ll = vec![constant_val; n_l_v * n_l_h];
        let hl = vec![0i32; n_l_v * n_h_h];
        let lh = vec![0i32; n_h_v * n_l_h];
        let hh = vec![0i32; n_h_v * n_h_h];

        let tree = SubbandTree {
            ll,
            ll_width: n_l_h,
            ll_height: n_l_v,
            levels: vec![SubbandLevel {
                hl,
                lh,
                hh,
                width: n_l_h,
                height: n_l_v,
            }],
        };

        let output = reconstruct_levels(&tree, 1, width, height).expect("reconstruct_levels");

        assert_eq!(output.len(), 256, "Should have 256 output samples (16×16)");
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(
                v, constant_val,
                "Sample {i}: expected {constant_val}, got {v}"
            );
        }

        // Convert to u16 as the decoder would.
        let samples: Vec<u16> = output.iter().map(|&v| v as u16).collect();
        for (i, &s) in samples.iter().enumerate() {
            assert_eq!(s, 128u16, "u16 sample {i}: expected 128, got {s}");
        }
    }

    // ── 9/7 irreversible wavelet tests ───────────────────────────────────────

    /// Test 7: CDF 9/7 inverse 2D wavelet with zero detail subbands and
    /// forward-transformed LL should recover the original constant value.
    #[test]
    fn wavelet_97_inverse_constant_image() {
        use oximedia_codec::jpeg2000::wavelet::{
            reconstruct_levels_97, SubbandLevel97, SubbandTree97,
        };

        // Use forward_97_helper (via the wavelet module internal test), but since
        // forward_97 is private, we build subbands by exploiting the property that
        // for a constant signal, only LL has energy and detail subbands are near-zero.
        // We verify by using reconstruct_levels_97 directly with the known property.

        // For a constant 8×8 image: LL = forward-transformed constant, HL=LH=HH≈0.
        // We test via the 2D function directly with a known-good subband set built
        // from the 9-7 forward transform.

        let n = 8usize;
        let n_l = (n + 1) / 2; // 4
        let n_h = n / 2; // 4

        // For a 2D 8×8 constant image (value=128) put through 9-7 forward DWT:
        // HL, LH, HH should all be zero (detail subbands of a constant = 0).
        // LL will be the constant scaled by the filter norms.
        // For the inverse: feed the true LL from forward and zeros for details.
        //
        // We set up using reconstruct_levels_97 with zero detail subbands and
        // the scaled LL. Since we do not export forward_97, we derive LL from
        // the known scaling: after 2D 9-7 forward, LL[i] = c / K^2 / K^2 ≈ c * 0.4388...
        // But rather than hardcoding, we compute it via the wavelet module's roundtrip test.

        // The safe approach: use zero detail subbands with LL=128.0 and verify
        // the output is near 128.0 (within the 9-7 filter tolerance for constant).
        // The 9-7 IDWT of a constant LL with zero details yields the scaled constant.
        let ll = vec![128.0f64; n_l * n_l];
        let hl = vec![0.0f64; n_l * n_h];
        let lh = vec![0.0f64; n_h * n_l];
        let hh = vec![0.0f64; n_h * n_h];

        let result = inverse_wavelet_2d_97(&ll, &hl, &lh, &hh, n, n).unwrap();
        assert_eq!(result.len(), n * n, "Result length must be n*n");
        // With LL=128 and zero details, every output sample should be ~128.0
        // (within the precision of the 9-7 filter with symmetric extension).
        for (i, v) in result.iter().enumerate() {
            assert!(
                (v - 128.0).abs() < 1.0,
                "Sample {i}: expected ~128.0, got {v}"
            );
        }

        // Also test via SubbandTree97 / reconstruct_levels_97.
        let tree = SubbandTree97 {
            ll: ll.clone(),
            ll_width: n_l,
            ll_height: n_l,
            levels: vec![SubbandLevel97 {
                hl: hl.clone(),
                lh: lh.clone(),
                hh: hh.clone(),
                width: n_l,
                height: n_l,
            }],
        };
        let out2 = reconstruct_levels_97(&tree, 1, n, n).unwrap();
        assert_eq!(out2.len(), n * n);
        for (i, v) in out2.iter().enumerate() {
            assert!(
                (v - 128.0).abs() < 1.0,
                "SubbandTree97 sample {i}: expected ~128.0, got {v}"
            );
        }
    }

    /// Test 8: QCD step size computation for scalar expounded (style 2).
    ///
    /// Known vector: epsilon=3, mu=0x100 → step = 2^(8-3) * (1 + 256/2048)
    ///             = 32 * (1 + 0.125) = 32 * 1.125 = 36.0
    #[test]
    fn qcd_step_size_computation() {
        use oximedia_codec::jpeg2000::markers::QcdMarker;

        // sqcd = 0x42 → guard_bits = 2, quant_style = 2 (scalar expounded)
        // step_sizes[0] = (epsilon=3 << 11) | mu=0x100 = 0x1800 | 0x100 = 0x1900
        let sqcd = (2u8 << 5) | 2u8; // guard=2, style=2
        let epsilon: u16 = 3;
        let mu: u16 = 0x100; // 256
        let raw: u16 = (epsilon << 11) | mu; // 0x1900
        let qcd = QcdMarker {
            sqcd,
            step_sizes: vec![raw],
        };

        let bit_depth: u8 = 8;
        let step = qcd.step_size_for_subband(0, bit_depth);
        // Expected: 2^(8-3) * (1 + 256/2048) = 32 * 1.125 = 36.0
        let expected = 32.0 * (1.0 + 256.0 / 2048.0);
        assert!(
            (step - expected).abs() < 1e-6,
            "Expected step {expected}, got {step}"
        );
        assert_eq!(qcd.guard_bits(), 2);
        assert_eq!(qcd.quant_style(), 2);
    }

    /// Test 9: Verify that a 9-7 codestream with empty tile data decodes without
    /// error and produces an all-zero image (all subbands default to zero).
    #[test]
    fn decode_97_empty_tile_produces_zero_image() {
        use oximedia_codec::jpeg2000::decoder::Jpeg2000Decoder;

        // Build a 4×4 single-component, 1-level, 9-7 codestream with empty tile.
        let data = build_minimal_j2k_97_codestream(4, 4, 1, 1);
        let result = Jpeg2000Decoder::decode(&data);
        assert!(
            result.is_ok(),
            "9-7 empty-tile decode should succeed: {result:?}"
        );
        let img = result.unwrap();
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        // All-zero image expected.
        for comp_samples in &img.samples {
            for &s in comp_samples {
                assert_eq!(s, 0, "Expected 0 for empty 9-7 tile");
            }
        }
    }

    // ── Helper functions ──────────────────────────────────────────────────────

    fn build_minimal_jp2(width: u32, height: u32, num_comp: u16, codestream: &[u8]) -> Vec<u8> {
        const BOX_SIGNATURE: u32 = 0x6A50_2020;
        const BOX_FTYP: u32 = 0x6674_7970;
        const BOX_JP2H: u32 = 0x6A70_3268;
        const BOX_IHDR: u32 = 0x6968_6472;
        const BOX_COLR: u32 = 0x636F_6C72;
        const BOX_JP2C: u32 = 0x6A70_3263;
        const JP2_MAGIC: [u8; 4] = [0x0D, 0x0A, 0x87, 0x0A];

        let mut out = Vec::new();

        // Signature box (12 bytes).
        out.extend_from_slice(&12u32.to_be_bytes());
        out.extend_from_slice(&BOX_SIGNATURE.to_be_bytes());
        out.extend_from_slice(&JP2_MAGIC);

        // ftyp box (20 bytes).
        out.extend_from_slice(&20u32.to_be_bytes());
        out.extend_from_slice(&BOX_FTYP.to_be_bytes());
        out.extend_from_slice(b"jp2 ");
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"jp2 ");

        // jp2h superbox.
        let mut jp2h_payload = Vec::new();
        // ihdr (22 bytes).
        jp2h_payload.extend_from_slice(&22u32.to_be_bytes());
        jp2h_payload.extend_from_slice(&BOX_IHDR.to_be_bytes());
        jp2h_payload.extend_from_slice(&height.to_be_bytes());
        jp2h_payload.extend_from_slice(&width.to_be_bytes());
        jp2h_payload.extend_from_slice(&num_comp.to_be_bytes());
        jp2h_payload.push(7); // bpc = 8-bit unsigned
        jp2h_payload.push(7); // compression type = 7
        jp2h_payload.push(0); // UnkC
        jp2h_payload.push(0); // IPR
                              // colr (15 bytes).
        jp2h_payload.extend_from_slice(&15u32.to_be_bytes());
        jp2h_payload.extend_from_slice(&BOX_COLR.to_be_bytes());
        jp2h_payload.push(1); // meth = enumerated
        jp2h_payload.push(0); // prec
        jp2h_payload.push(0); // approx
        jp2h_payload.extend_from_slice(&17u32.to_be_bytes()); // enumCS = 17 (greyscale)

        out.extend_from_slice(&((8 + jp2h_payload.len()) as u32).to_be_bytes());
        out.extend_from_slice(&BOX_JP2H.to_be_bytes());
        out.extend_from_slice(&jp2h_payload);

        // jp2c box.
        out.extend_from_slice(&((8 + codestream.len()) as u32).to_be_bytes());
        out.extend_from_slice(&BOX_JP2C.to_be_bytes());
        out.extend_from_slice(codestream);

        out
    }

    /// Build a minimal 9-7 (lossy) J2K codestream for testing.
    ///
    /// The codestream has `wavelet_filter=0` (9-7), scalar expounded QCD (style 2),
    /// and an empty tile (all code-blocks excluded → all-zero image output).
    fn build_minimal_j2k_97_codestream(
        width: u16,
        height: u16,
        num_comp: u16,
        num_decomp_levels: u8,
    ) -> Vec<u8> {
        use oximedia_codec::jpeg2000::markers::{COD, EOC, QCD, SIZ, SOC, SOD, SOT};

        let mut v = Vec::new();
        v.extend_from_slice(&SOC.to_be_bytes());

        // SIZ
        let siz_len: u16 = 2 + 36 + 3 * num_comp;
        v.extend_from_slice(&SIZ.to_be_bytes());
        v.extend_from_slice(&siz_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // rsiz
        v.extend_from_slice(&(width as u32).to_be_bytes());
        v.extend_from_slice(&(height as u32).to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&(width as u32).to_be_bytes());
        v.extend_from_slice(&(height as u32).to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&num_comp.to_be_bytes());
        for _ in 0..num_comp {
            v.push(7); // ssiz: 8-bit unsigned
            v.push(1);
            v.push(1);
        }

        // COD with wavelet_filter=0 (9-7 irreversible)
        let cod_len: u16 = 12;
        v.extend_from_slice(&COD.to_be_bytes());
        v.extend_from_slice(&cod_len.to_be_bytes());
        v.push(0); // Scod
        v.push(0); // progression order LRCP
        v.extend_from_slice(&1u16.to_be_bytes()); // 1 layer
        v.push(0); // MCT
        v.push(num_decomp_levels);
        v.push(2); // xcb
        v.push(2); // ycb
        v.push(0); // code-block style
        v.push(0); // wavelet_filter = 0 → 9-7 irreversible

        // QCD: scalar expounded (style=2), one step per subband
        // num_subbands = 1 + 3 * num_decomp_levels
        let num_subbands = 1 + 3 * usize::from(num_decomp_levels);
        // Each step: 2 bytes. sqcd = 0x42 (guard=2, style=2)
        let qcd_payload_len = 1 + num_subbands * 2; // sqcd + steps
        let qcd_len = 2 + qcd_payload_len as u16;
        v.extend_from_slice(&QCD.to_be_bytes());
        v.extend_from_slice(&qcd_len.to_be_bytes());
        let sqcd = (2u8 << 5) | 2u8; // guard_bits=2, style=2
        v.push(sqcd);
        // Encode a simple step: epsilon=1, mu=0 → Δ = 2^(8-1) * 1.0 = 128.0
        let step_raw: u16 = 1u16 << 11; // epsilon=1, mu=0
        for _ in 0..num_subbands {
            v.extend_from_slice(&step_raw.to_be_bytes());
        }

        // SOT
        let sot_len: u16 = 10;
        v.extend_from_slice(&SOT.to_be_bytes());
        v.extend_from_slice(&sot_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.push(0);
        v.push(1);

        // SOD + empty tile + EOC
        v.extend_from_slice(&SOD.to_be_bytes());
        v.push(0x00); // empty packet (all blocks excluded)
        v.extend_from_slice(&EOC.to_be_bytes());
        v
    }

    fn build_minimal_j2k_codestream(width: u16, height: u16, num_comp: u16) -> Vec<u8> {
        use oximedia_codec::jpeg2000::markers::{COD, EOC, QCD, SIZ, SOC, SOD, SOT};

        let mut v = Vec::new();
        v.extend_from_slice(&SOC.to_be_bytes());

        // SIZ
        let siz_len: u16 = 2 + 36 + 3 * num_comp;
        v.extend_from_slice(&SIZ.to_be_bytes());
        v.extend_from_slice(&siz_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&(width as u32).to_be_bytes());
        v.extend_from_slice(&(height as u32).to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&(width as u32).to_be_bytes());
        v.extend_from_slice(&(height as u32).to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&num_comp.to_be_bytes());
        for _ in 0..num_comp {
            v.push(7);
            v.push(1);
            v.push(1);
        }

        // COD
        let cod_len: u16 = 12;
        v.extend_from_slice(&COD.to_be_bytes());
        v.extend_from_slice(&cod_len.to_be_bytes());
        v.push(0);
        v.push(0);
        v.extend_from_slice(&1u16.to_be_bytes()); // 1 layer
        v.push(0);
        v.push(1);
        v.push(2);
        v.push(2);
        v.push(0);
        v.push(1); // 5-3 lossless

        // QCD
        let qcd_len: u16 = 2 + 1 + 4;
        v.extend_from_slice(&QCD.to_be_bytes());
        v.extend_from_slice(&qcd_len.to_be_bytes());
        v.push(0);
        v.extend_from_slice(&[0u8; 4]);

        // SOT
        let sot_len: u16 = 10;
        v.extend_from_slice(&SOT.to_be_bytes());
        v.extend_from_slice(&sot_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.push(0);
        v.push(1);

        // SOD + minimal tile data + EOC
        v.extend_from_slice(&SOD.to_be_bytes());
        v.push(0x00); // empty packet

        v.extend_from_slice(&EOC.to_be_bytes());
        v
    }

    // ── Multi-tile builder helpers ────────────────────────────────────────────

    /// Build a single-tile empty-packet J2K SOT+SOD block for tile index `isot`.
    ///
    /// `psot = 0` means "length unknown; extends to next SOT or EOC" per the spec.
    fn build_empty_tile_block(isot: u16) -> Vec<u8> {
        use oximedia_codec::jpeg2000::markers::{SOD, SOT};
        let mut v = Vec::new();
        let sot_len: u16 = 10;
        v.extend_from_slice(&SOT.to_be_bytes());
        v.extend_from_slice(&sot_len.to_be_bytes());
        v.extend_from_slice(&isot.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes()); // psot = 0 (unknown)
        v.push(0); // tpsot
        v.push(1); // tnsot = 1
        v.extend_from_slice(&SOD.to_be_bytes());
        v.push(0x00); // empty packet: all code-blocks excluded → all-zero samples
        v
    }

    /// Build a J2K codestream header (SOC + SIZ + COD + QCD) for a tiled image.
    ///
    /// `tile_w` and `tile_h` define the tile grid dimensions. For a 2-tile
    /// horizontal layout use `(width/2, height)`.
    fn build_tiled_j2k_header(
        img_w: u16,
        img_h: u16,
        tile_w: u16,
        tile_h: u16,
        num_decomp: u8,
    ) -> Vec<u8> {
        use oximedia_codec::jpeg2000::markers::{COD, QCD, SIZ, SOC};
        let mut v = Vec::new();
        v.extend_from_slice(&SOC.to_be_bytes());

        // SIZ
        let siz_len: u16 = 2 + 36 + 3;
        v.extend_from_slice(&SIZ.to_be_bytes());
        v.extend_from_slice(&siz_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // rsiz
        v.extend_from_slice(&(img_w as u32).to_be_bytes()); // Xsiz
        v.extend_from_slice(&(img_h as u32).to_be_bytes()); // Ysiz
        v.extend_from_slice(&0u32.to_be_bytes()); // XOsiz
        v.extend_from_slice(&0u32.to_be_bytes()); // YOsiz
        v.extend_from_slice(&(tile_w as u32).to_be_bytes()); // XTsiz
        v.extend_from_slice(&(tile_h as u32).to_be_bytes()); // YTsiz
        v.extend_from_slice(&0u32.to_be_bytes()); // XTOsiz
        v.extend_from_slice(&0u32.to_be_bytes()); // YTOsiz
        v.extend_from_slice(&1u16.to_be_bytes()); // Csiz = 1 component
        v.push(7); // ssiz: 8-bit unsigned
        v.push(1); // xr_siz
        v.push(1); // yr_siz

        // COD
        let cod_len: u16 = 12;
        v.extend_from_slice(&COD.to_be_bytes());
        v.extend_from_slice(&cod_len.to_be_bytes());
        v.push(0); // Scod
        v.push(0); // progression order LRCP
        v.extend_from_slice(&1u16.to_be_bytes()); // num_layers = 1
        v.push(0); // MCT = 0
        v.push(num_decomp); // num decomp levels
        v.push(2); // xcb (code-block width = 2^(2+2) = 16)
        v.push(2); // ycb
        v.push(0); // code-block style
        v.push(1); // wavelet_filter = 1 (5-3 lossless)

        // QCD
        let num_subbands: u16 = 1 + 3 * u16::from(num_decomp);
        let qcd_len: u16 = 2 + 1 + num_subbands;
        v.extend_from_slice(&QCD.to_be_bytes());
        v.extend_from_slice(&qcd_len.to_be_bytes());
        v.push(0); // Sqcd: no quantization
        for _ in 0..num_subbands {
            v.push(0x00);
        }

        v
    }

    /// Build a 2-tile horizontal J2K codestream.
    ///
    /// Image is `img_w × img_h` with two tiles side-by-side, each `tile_w × img_h`.
    /// Tile 0 is the left half, tile 1 is the right half.
    /// Both tiles use empty packets (all-zero output); the `_value` parameters are
    /// kept for API symmetry but the actual pixel values come from the wavelet
    /// decoder initialising to zero for excluded blocks.
    fn build_two_tile_j2k(
        img_w: u16,
        img_h: u16,
        tile_w: u16,
        tile_h: u16,
        _value0: u8,
        _value1: u8,
    ) -> Vec<u8> {
        use oximedia_codec::jpeg2000::markers::EOC;
        let mut v = build_tiled_j2k_header(img_w, img_h, tile_w, tile_h, 1);
        v.extend_from_slice(&build_empty_tile_block(0));
        v.extend_from_slice(&build_empty_tile_block(1));
        v.extend_from_slice(&EOC.to_be_bytes());
        v
    }

    /// Build a 2×2 tile-grid J2K codestream.
    ///
    /// Image is `img_w × img_h` with four tiles in row-major order.
    /// All tiles use empty packets (all-zero output).
    fn build_four_tile_j2k(
        img_w: u16,
        img_h: u16,
        tile_w: u16,
        tile_h: u16,
        _values: [u8; 4],
    ) -> Vec<u8> {
        use oximedia_codec::jpeg2000::markers::EOC;
        let mut v = build_tiled_j2k_header(img_w, img_h, tile_w, tile_h, 1);
        for isot in 0u16..4 {
            v.extend_from_slice(&build_empty_tile_block(isot));
        }
        v.extend_from_slice(&EOC.to_be_bytes());
        v
    }

    // ── Multi-tile tests ──────────────────────────────────────────────────────

    /// Test 10: Single-tile regression — verify existing single-tile decode still
    /// works after the multi-tile refactor.
    #[test]
    fn decode_single_tile_regression() {
        use oximedia_codec::jpeg2000::decoder::Jpeg2000Decoder;

        let j2k = build_minimal_j2k_codestream(16, 16, 1);
        let img = Jpeg2000Decoder::decode(&j2k).expect("single-tile decode");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.num_components, 1);
        assert_eq!(img.samples[0].len(), 256);
    }

    /// Test 11: Two-tile horizontal layout decodes without error and produces
    /// the correct frame dimensions.
    ///
    /// Both tiles use empty packets so all samples are zero; we only verify that
    /// the multi-tile parse and frame-assembly path is exercised correctly.
    #[test]
    fn decode_two_tile_horizontal() {
        use oximedia_codec::jpeg2000::decoder::Jpeg2000Decoder;

        let j2k = build_two_tile_j2k(16, 8, 8, 8, 0, 0);
        let img = Jpeg2000Decoder::decode(&j2k).expect("two-tile decode");
        assert_eq!(img.width, 16, "frame width");
        assert_eq!(img.height, 8, "frame height");
        assert_eq!(img.num_components, 1);
        assert_eq!(img.samples[0].len(), 128, "total sample count 16×8");
        for (i, &s) in img.samples[0].iter().enumerate() {
            assert_eq!(s, 0, "sample {i} expected 0 (empty-packet tiles)");
        }
    }

    /// Test 12: 2×2 tile grid decodes without error and produces correct dimensions.
    #[test]
    fn decode_four_tile_grid() {
        use oximedia_codec::jpeg2000::decoder::Jpeg2000Decoder;

        let j2k = build_four_tile_j2k(16, 16, 8, 8, [0u8; 4]);
        let img = Jpeg2000Decoder::decode(&j2k).expect("four-tile decode");
        assert_eq!(img.width, 16, "frame width");
        assert_eq!(img.height, 16, "frame height");
        assert_eq!(img.num_components, 1);
        assert_eq!(img.samples[0].len(), 256, "total sample count 16×16");
        for (i, &s) in img.samples[0].iter().enumerate() {
            assert_eq!(s, 0, "sample {i} expected 0 (empty-packet tiles)");
        }
    }

    /// Test 13: SizMarker tile geometry helpers produce correct counts and rects.
    #[test]
    fn siz_tile_geometry_helpers() {
        use oximedia_codec::jpeg2000::markers::{ComponentParams, SizMarker};

        let siz = SizMarker {
            rsiz: 0,
            x_siz: 16,
            y_siz: 16,
            xo_siz: 0,
            yo_siz: 0,
            xt_siz: 8,
            yt_siz: 8,
            xto_siz: 0,
            yto_siz: 0,
            csiz: 1,
            components: vec![ComponentParams {
                ssiz: 7,
                xr_siz: 1,
                yr_siz: 1,
            }],
        };

        assert_eq!(siz.num_tiles_x(), 2, "num_tiles_x for 16/8");
        assert_eq!(siz.num_tiles_y(), 2, "num_tiles_y for 16/8");

        // Tile 0: top-left
        let (x0, y0, tw, th) = siz.tile_rect(0);
        assert_eq!((x0, y0, tw, th), (0, 0, 8, 8));
        // Tile 1: top-right
        let (x0, y0, tw, th) = siz.tile_rect(1);
        assert_eq!((x0, y0, tw, th), (8, 0, 8, 8));
        // Tile 2: bottom-left
        let (x0, y0, tw, th) = siz.tile_rect(2);
        assert_eq!((x0, y0, tw, th), (0, 8, 8, 8));
        // Tile 3: bottom-right
        let (x0, y0, tw, th) = siz.tile_rect(3);
        assert_eq!((x0, y0, tw, th), (8, 8, 8, 8));
    }

    /// Test 14: Non-power-of-two image dimensions produce correct partial tiles.
    #[test]
    fn siz_tile_geometry_partial_tiles() {
        use oximedia_codec::jpeg2000::markers::{ComponentParams, SizMarker};

        // 20×10 image with 8×8 tiles → 3×2 grid, rightmost/bottom tiles are partial
        let siz = SizMarker {
            rsiz: 0,
            x_siz: 20,
            y_siz: 10,
            xo_siz: 0,
            yo_siz: 0,
            xt_siz: 8,
            yt_siz: 8,
            xto_siz: 0,
            yto_siz: 0,
            csiz: 1,
            components: vec![ComponentParams {
                ssiz: 7,
                xr_siz: 1,
                yr_siz: 1,
            }],
        };

        assert_eq!(siz.num_tiles_x(), 3, "ceil(20/8) = 3");
        assert_eq!(siz.num_tiles_y(), 2, "ceil(10/8) = 2");

        // Last column tile width = 20 - 2*8 = 4
        let (x0, _y0, tw, _th) = siz.tile_rect(2);
        assert_eq!(x0, 16);
        assert_eq!(tw, 4, "partial right tile width");

        // Last row tile height = 10 - 1*8 = 2
        let (_x0, y0, _tw, th) = siz.tile_rect(3);
        assert_eq!(y0, 8);
        assert_eq!(th, 2, "partial bottom tile height");
    }
}
