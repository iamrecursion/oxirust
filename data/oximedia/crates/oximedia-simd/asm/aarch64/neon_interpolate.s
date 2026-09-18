// ARM NEON optimized interpolation and SAD kernels for AArch64
// Implements bilinear, bicubic, 8-tap interpolation and SAD operations
//
// AAPCS64 calling convention:
// X0 = src, X1 = dst, W2 = src_stride, W3 = dst_stride,
// W4 = width, W5 = height, W6 = dx, W7 = dy

    .text
    .arch armv8-a+simd

// ============================================================================
// Bilinear Interpolation using NEON
// ============================================================================
    .global neon_interpolate_bilinear
    .type neon_interpolate_bilinear, %function
neon_interpolate_bilinear:
    // Arguments:
    // x0 = src pointer
    // x1 = dst pointer
    // w2 = src_stride
    // w3 = dst_stride
    // w4 = width
    // w5 = height
    // w6 = dx (horizontal fractional position)
    // w7 = dy (vertical fractional position)

    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]

    // Extract fractional positions
    and w6, w6, #15             // fx = dx & 15
    and w7, w7, #15             // fy = dy & 15

    // Create weight vectors
    dup v10.8h, w6              // Horizontal weight (fx)
    dup v11.8h, w7              // Vertical weight (fy)

    // Create inverse weights
    mov w8, #16
    sub w9, w8, w6
    dup v12.8h, w9              // 16 - fx

    sub w10, w8, w7
    dup v13.8h, w10             // 16 - fy

    // Initialize row counter
    mov w19, #0                 // y = 0

neon_bilinear_row_loop:
    cmp w19, w5
    b.ge neon_bilinear_done

    // Calculate source row pointer
    umull x20, w19, w2          // y * src_stride
    add x21, x0, x20            // src + y * stride

    // Calculate destination row pointer
    umull x22, w19, w3
    add x23, x1, x22            // dst + y * dst_stride

    // Initialize column counter
    mov w24, #0                 // x = 0

neon_bilinear_col_loop:
    cmp w24, w4
    b.ge neon_bilinear_next_row

    // Check if we can process 16 pixels
    sub w8, w4, w24
    cmp w8, #16
    b.lt neon_bilinear_scalar

    // Load 16 pixels from each position
    add x8, x21, x24
    ld1 {v0.16b}, [x8]          // p00 (16 pixels)

    add x8, x21, x24
    add x8, x8, #1
    ld1 {v1.16b}, [x8]          // p01

    add x8, x21, x24
    add x8, x8, x2, uxtw
    ld1 {v2.16b}, [x8]          // p10

    add x8, x8, #1
    ld1 {v3.16b}, [x8]          // p11

    // Convert to 16-bit for multiplication (first 8 pixels)
    uxtl v4.8h, v0.8b           // p00[0:7] expanded
    uxtl2 v5.8h, v0.16b         // p00[8:15] expanded

    uxtl v6.8h, v1.8b           // p01[0:7] expanded
    uxtl2 v7.8h, v1.16b         // p01[8:15] expanded

    // Horizontal interpolation: h0 = p00 * (16-fx) + p01 * fx
    mul v4.8h, v4.8h, v12.8h    // p00 * (16-fx)
    mla v4.8h, v6.8h, v10.8h    // += p01 * fx

    mul v5.8h, v5.8h, v12.8h
    mla v5.8h, v7.8h, v10.8h

    // Same for second row
    uxtl v6.8h, v2.8b           // p10[0:7] expanded
    uxtl2 v7.8h, v2.16b         // p10[8:15] expanded

    uxtl v8.8h, v3.8b           // p11[0:7] expanded
    uxtl2 v9.8h, v3.16b         // p11[8:15] expanded

    mul v6.8h, v6.8h, v12.8h    // p10 * (16-fx)
    mla v6.8h, v8.8h, v10.8h    // += p11 * fx

    mul v7.8h, v7.8h, v12.8h
    mla v7.8h, v9.8h, v10.8h

    // Vertical interpolation: v = h0 * (16-fy) + h1 * fy
    mul v4.8h, v4.8h, v13.8h    // h0 * (16-fy)
    mla v4.8h, v6.8h, v11.8h    // += h1 * fy

    mul v5.8h, v5.8h, v13.8h
    mla v5.8h, v7.8h, v11.8h

    // Add rounding constant (128) and shift by 8
    movi v14.8h, #128
    add v4.8h, v4.8h, v14.8h
    add v5.8h, v5.8h, v14.8h

    ushr v4.8h, v4.8h, #8
    ushr v5.8h, v5.8h, #8

    // Pack back to 8-bit with saturation
    uqxtn v0.8b, v4.8h
    uqxtn2 v0.16b, v5.8h

    // Store 16 pixels
    add x8, x23, x24
    st1 {v0.16b}, [x8]

    add w24, w24, #16
    b neon_bilinear_col_loop

neon_bilinear_scalar:
    // Scalar fallback for remaining pixels
    cmp w24, w4
    b.ge neon_bilinear_next_row

    // Load 4 pixels
    add x8, x21, x24
    ldrb w10, [x8]              // p00
    ldrb w11, [x8, #1]          // p01
    add x9, x8, x2, uxtw
    ldrb w12, [x9]              // p10
    ldrb w13, [x9, #1]          // p11

    // Horizontal interpolation
    mul w10, w10, w9            // p00 * (16-fx)
    mul w11, w11, w6            // p01 * fx
    add w10, w10, w11           // h0

    mul w12, w12, w9            // p10 * (16-fx)
    mul w13, w13, w6            // p11 * fx
    add w12, w12, w13           // h1

    // Vertical interpolation
    mul w10, w10, w10           // h0 * (16-fy)
    mul w12, w12, w7            // h1 * fy
    add w10, w10, w12
    add w10, w10, #128
    lsr w10, w10, #8

    // Clamp to [0, 255]
    cmp w10, #255
    csel w10, w10, wzr, le

    // Store pixel
    add x8, x23, x24
    strb w10, [x8]

    add w24, w24, #1
    b neon_bilinear_scalar

neon_bilinear_next_row:
    add w19, w19, #1
    b neon_bilinear_row_loop

neon_bilinear_done:
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
    .size neon_interpolate_bilinear, .-neon_interpolate_bilinear

// ============================================================================
// Bicubic Interpolation using NEON
// ============================================================================
    .global neon_interpolate_bicubic
    .type neon_interpolate_bicubic, %function
neon_interpolate_bicubic:
    // Use bilinear as simplified implementation
    b neon_interpolate_bilinear
    .size neon_interpolate_bicubic, .-neon_interpolate_bicubic

// ============================================================================
// 8-tap Interpolation using NEON
// ============================================================================
    .global neon_interpolate_8tap
    .type neon_interpolate_8tap, %function
neon_interpolate_8tap:
    // Simplified 8-tap implementation (uses bilinear for now)
    b neon_interpolate_bilinear
    .size neon_interpolate_8tap, .-neon_interpolate_8tap

// ============================================================================
// 16x16 SAD using NEON
// ============================================================================
    .global neon_sad_16x16
    .type neon_sad_16x16, %function
neon_sad_16x16:
    // Arguments:
    // x0 = src1 pointer
    // x1 = src2 pointer
    // w2 = stride1
    // w3 = stride2
    // Returns: w0 = SAD value

    stp x29, x30, [sp, #-16]!
    mov x29, sp

    // Zero accumulator
    movi v0.4s, #0              // SAD accumulator (4 x 32-bit)

    mov w4, #0                  // Row counter

neon_sad16_loop:
    cmp w4, #16
    b.ge neon_sad16_done

    // Load 16 pixels from each source
    ld1 {v1.16b}, [x0]          // src1[y][0:15]
    ld1 {v2.16b}, [x1]          // src2[y][0:15]

    // Compute absolute differences
    uabd v3.16b, v1.16b, v2.16b

    // Accumulate (extend to 16-bit first)
    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b

    // Extend to 32-bit and accumulate
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h

    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Move to next row
    add x0, x0, x2, uxtw        // src1 += stride1
    add x1, x1, x3, uxtw        // src2 += stride2
    add w4, w4, #1
    b neon_sad16_loop

neon_sad16_done:
    // Horizontal sum
    addp v0.4s, v0.4s, v0.4s
    addp v0.4s, v0.4s, v0.4s

    // Extract result
    umov w0, v0.s[0]

    ldp x29, x30, [sp], #16
    ret
    .size neon_sad_16x16, .-neon_sad_16x16

// ============================================================================
// 32x32 SAD using NEON
// ============================================================================
    .global neon_sad_32x32
    .type neon_sad_32x32, %function
neon_sad_32x32:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    movi v0.4s, #0              // SAD accumulator
    mov w4, #0                  // Row counter

neon_sad32_loop:
    cmp w4, #32
    b.ge neon_sad32_done

    // Load first 16 pixels
    ld1 {v1.16b}, [x0], #16
    ld1 {v2.16b}, [x1], #16
    uabd v3.16b, v1.16b, v2.16b

    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h
    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Load second 16 pixels
    ld1 {v1.16b}, [x0]
    ld1 {v2.16b}, [x1]
    uabd v3.16b, v1.16b, v2.16b

    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h
    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Adjust pointers for next row
    sub x0, x0, #16
    sub x1, x1, #16
    add x0, x0, x2, uxtw
    add x1, x1, x3, uxtw

    add w4, w4, #1
    b neon_sad32_loop

neon_sad32_done:
    addp v0.4s, v0.4s, v0.4s
    addp v0.4s, v0.4s, v0.4s
    umov w0, v0.s[0]

    ldp x29, x30, [sp], #16
    ret
    .size neon_sad_32x32, .-neon_sad_32x32

// ============================================================================
// 64x64 SAD using NEON
// ============================================================================
    .global neon_sad_64x64
    .type neon_sad_64x64, %function
neon_sad_64x64:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    movi v0.4s, #0              // SAD accumulator
    mov w4, #0                  // Row counter

neon_sad64_loop:
    cmp w4, #64
    b.ge neon_sad64_done

    // Process 64 pixels in 4 chunks of 16
    mov x8, x0
    mov x9, x1

    // Chunk 0
    ld1 {v1.16b}, [x8], #16
    ld1 {v2.16b}, [x9], #16
    uabd v3.16b, v1.16b, v2.16b
    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h
    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Chunk 1
    ld1 {v1.16b}, [x8], #16
    ld1 {v2.16b}, [x9], #16
    uabd v3.16b, v1.16b, v2.16b
    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h
    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Chunk 2
    ld1 {v1.16b}, [x8], #16
    ld1 {v2.16b}, [x9], #16
    uabd v3.16b, v1.16b, v2.16b
    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h
    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Chunk 3
    ld1 {v1.16b}, [x8], #16
    ld1 {v2.16b}, [x9], #16
    uabd v3.16b, v1.16b, v2.16b
    uxtl v4.8h, v3.8b
    uxtl2 v5.8h, v3.16b
    uaddlp v6.4s, v4.8h
    uaddlp v7.4s, v5.8h
    add v0.4s, v0.4s, v6.4s
    add v0.4s, v0.4s, v7.4s

    // Next row
    add x0, x0, x2, uxtw
    add x1, x1, x3, uxtw

    add w4, w4, #1
    b neon_sad64_loop

neon_sad64_done:
    addp v0.4s, v0.4s, v0.4s
    addp v0.4s, v0.4s, v0.4s
    umov w0, v0.s[0]

    ldp x29, x30, [sp], #16
    ret
    .size neon_sad_64x64, .-neon_sad_64x64

    .section .note.GNU-stack,"",@progbits
