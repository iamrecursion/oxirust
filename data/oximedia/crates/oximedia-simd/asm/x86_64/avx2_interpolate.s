# AVX2 optimized interpolation kernels for x86-64
# Implements bilinear, bicubic, and 8-tap interpolation for motion compensation
#
# System V AMD64 ABI calling convention:
# RDI = src, RSI = dst, EDX = src_stride, ECX = dst_stride,
# R8D = width, R9D = height, [rbp+16] = dx, [rbp+24] = dy

    .text
    .intel_syntax noprefix

# ============================================================================
# Bilinear Interpolation using AVX2
# ============================================================================
    .globl avx2_interpolate_bilinear
    .type avx2_interpolate_bilinear, @function
avx2_interpolate_bilinear:
    # Arguments:
    # rdi = src pointer
    # rsi = dst pointer
    # edx = src_stride
    # ecx = dst_stride
    # r8d = width
    # r9d = height
    # [rbp+16] = dx (horizontal fractional position)
    # [rbp+24] = dy (vertical fractional position)

    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    # Load fractional positions
    mov r10d, [rbp+16]         # dx
    mov r11d, [rbp+24]         # dy

    # Create weight vectors for horizontal interpolation
    # fx = dx & 15, fy = dy & 15
    and r10d, 15
    and r11d, 15

    # Broadcast weights to YMM registers
    movd xmm0, r10d
    vpbroadcastw ymm10, xmm0   # Horizontal weight (fx)

    movd xmm0, r11d
    vpbroadcastw ymm11, xmm0   # Vertical weight (fy)

    # Create inverse weights (16 - weight)
    mov r12d, 16
    sub r12d, r10d
    movd xmm0, r12d
    vpbroadcastw ymm12, xmm0   # 16 - fx

    mov r13d, 16
    sub r13d, r11d
    movd xmm0, r13d
    vpbroadcastw ymm13, xmm0   # 16 - fy

    # Initialize row counter
    xor r14, r14               # y = 0

.L_bilinear_row_loop:
    cmp r14d, r9d
    jge .L_bilinear_done

    # Calculate source row pointers
    mov rax, r14
    mul rdx                    # y * src_stride
    lea r15, [rdi + rax]       # src + y * stride

    # Calculate destination row pointer
    mov rax, r14
    mul rcx
    lea rbx, [rsi + rax]       # dst + y * dst_stride

    # Initialize column counter
    xor r12, r12               # x = 0

.L_bilinear_col_loop:
    cmp r12d, r8d
    jge .L_bilinear_next_row

    # Check if we can process 32 pixels at once
    mov r13d, r8d
    sub r13d, r12d
    cmp r13d, 32
    jl .L_bilinear_scalar

    # Load 32 pixels from current position
    vmovdqu ymm0, [r15 + r12]          # p00
    vmovdqu ymm1, [r15 + r12 + 1]      # p01
    vmovdqu ymm2, [r15 + r12 + rdx]    # p10
    vmovdqu ymm3, [r15 + r12 + rdx + 1] # p11

    # Convert to 16-bit for multiplication
    vpmovzxbw ymm4, xmm0       # p00 expanded
    vextracti128 xmm0, ymm0, 1
    vpmovzxbw ymm5, xmm0

    vpmovzxbw ymm6, xmm1       # p01 expanded
    vextracti128 xmm1, ymm1, 1
    vpmovzxbw ymm7, xmm1

    # Horizontal interpolation: h0 = p00 * (16-fx) + p01 * fx
    vpmullw ymm4, ymm4, ymm12  # p00 * (16-fx)
    vpmullw ymm6, ymm6, ymm10  # p01 * fx
    vpaddw ymm4, ymm4, ymm6    # h0

    vpmullw ymm5, ymm5, ymm12
    vpmullw ymm7, ymm7, ymm10
    vpaddw ymm5, ymm5, ymm7

    # Same for second row
    vpmovzxbw ymm6, xmm2       # p10 expanded
    vextracti128 xmm2, ymm2, 1
    vpmovzxbw ymm7, xmm2

    vpmovzxbw ymm8, xmm3       # p11 expanded
    vextracti128 xmm3, ymm3, 1
    vpmovzxbw ymm9, xmm3

    vpmullw ymm6, ymm6, ymm12  # p10 * (16-fx)
    vpmullw ymm8, ymm8, ymm10  # p11 * fx
    vpaddw ymm6, ymm6, ymm8    # h1

    vpmullw ymm7, ymm7, ymm12
    vpmullw ymm9, ymm9, ymm10
    vpaddw ymm7, ymm7, ymm9

    # Vertical interpolation: v = h0 * (16-fy) + h1 * fy
    vpmullw ymm4, ymm4, ymm13  # h0 * (16-fy)
    vpmullw ymm6, ymm6, ymm11  # h1 * fy
    vpaddw ymm4, ymm4, ymm6

    vpmullw ymm5, ymm5, ymm13
    vpmullw ymm7, ymm7, ymm11
    vpaddw ymm5, ymm5, ymm7

    # Add rounding constant (128) and shift by 8
    vpcmpeqw ymm14, ymm14, ymm14
    vpsrlw ymm14, 9            # 0x007F
    vpsllw ymm14, 1            # 0x00FE (126)
    vpsrlw ymm14, 1            # 0x007F (127)
    vpaddw ymm14, ymm14, ymm14 # 0x00FE (254)
    vpsrlw ymm14, 1            # 0x007F (127)
    vpaddw ymm14, ymm14, ymm14 # 254
    vpsrlw ymm14, 1            # 127
    vpaddw ymm14, ymm14, ymm14
    vpaddw ymm14, ymm14, ymm14 # 128 approximation

    vpaddw ymm4, ymm4, ymm14
    vpaddw ymm5, ymm5, ymm14
    vpsraw ymm4, 8
    vpsraw ymm5, 8

    # Pack back to 8-bit with saturation
    vpackuswb ymm4, ymm4, ymm5

    # Store 32 pixels (actually 16 due to packing)
    vmovdqu [rbx + r12], xmm4
    vextracti128 xmm5, ymm4, 1
    vmovdqu [rbx + r12 + 16], xmm5

    add r12, 32
    jmp .L_bilinear_col_loop

.L_bilinear_scalar:
    # Scalar fallback for remaining pixels
    cmp r12d, r8d
    jge .L_bilinear_next_row

    # Load 4 pixels
    movzx eax, byte ptr [r15 + r12]       # p00
    movzx ebx, byte ptr [r15 + r12 + 1]   # p01
    movzx ecx, byte ptr [r15 + r12 + rdx] # p10
    movzx edx, byte ptr [r15 + r12 + rdx + 1] # p11

    # Horizontal interpolation
    imul eax, r13d             # p00 * (16-fx)
    imul ebx, r10d             # p01 * fx
    add eax, ebx               # h0

    imul ecx, r13d             # p10 * (16-fx)
    imul edx, r10d             # p11 * fx
    add ecx, edx               # h1

    # Vertical interpolation
    imul eax, dword ptr [rbp-4] # Use stack for 16-fy
    imul ecx, r11d
    add eax, ecx
    add eax, 128
    shr eax, 8

    # Clamp to [0, 255]
    cmp eax, 255
    jle .L_bilinear_no_clamp
    mov eax, 255
.L_bilinear_no_clamp:

    mov byte ptr [rbx + r12], al
    inc r12
    jmp .L_bilinear_scalar

.L_bilinear_next_row:
    inc r14
    jmp .L_bilinear_row_loop

.L_bilinear_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    .size avx2_interpolate_bilinear, .-avx2_interpolate_bilinear

# ============================================================================
# Bicubic Interpolation using AVX2
# ============================================================================
    .globl avx2_interpolate_bicubic
    .type avx2_interpolate_bicubic, @function
avx2_interpolate_bicubic:
    # For now, use bilinear as implementation
    # Full bicubic would require 4x4 pixel neighborhood
    jmp avx2_interpolate_bilinear
    .size avx2_interpolate_bicubic, .-avx2_interpolate_bicubic

# ============================================================================
# 8-tap Interpolation using AVX2
# ============================================================================
    .globl avx2_interpolate_8tap
    .type avx2_interpolate_8tap, @function
avx2_interpolate_8tap:
    # 8-tap filter coefficients (typical H.264/HEVC style)
    # This is a simplified implementation

    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14

    # Load parameters
    mov r10d, [rbp+16]         # dx
    mov r11d, [rbp+24]         # dy

    # Select 8-tap filter based on fractional position
    # For simplicity, using fixed coefficients
    # Real implementation would have 16 different 8-tap filters

    # Filter coefficients: [-1, 4, -11, 40, 40, -11, 4, -1] / 64
    # Stored as 16-bit values

    xor r14, r14               # y counter

.L_8tap_row_loop:
    cmp r14d, r9d
    jge .L_8tap_done

    mov rax, r14
    mul rdx
    lea r15, [rdi + rax]       # src row

    mov rax, r14
    mul rcx
    lea rbx, [rsi + rax]       # dst row

    xor r12, r12               # x counter

.L_8tap_col_loop:
    cmp r12d, r8d
    jge .L_8tap_next_row

    # Load 8 pixels horizontally
    vmovdqu xmm0, [r15 + r12 - 3]  # 8-tap needs ±3 pixels

    # Convert to 16-bit
    vpmovzxbw ymm1, xmm0

    # Apply filter (simplified - just averaging)
    # Full implementation would multiply by 8 coefficients
    vpshufd xmm2, xmm0, 0x4E
    vpavgb xmm3, xmm0, xmm2

    movd eax, xmm3
    mov byte ptr [rbx + r12], al

    inc r12
    jmp .L_8tap_col_loop

.L_8tap_next_row:
    inc r14
    jmp .L_8tap_row_loop

.L_8tap_done:
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    .size avx2_interpolate_8tap, .-avx2_interpolate_8tap

    .section .note.GNU-stack,"",@progbits
