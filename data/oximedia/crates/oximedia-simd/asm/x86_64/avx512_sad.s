# AVX-512 optimized Sum of Absolute Differences (SAD) for x86-64
# Implements SAD for 16x16, 32x32, and 64x64 blocks
#
# AVX-512 provides 32-wide operations and additional registers
# System V AMD64 ABI: RDI = src1, RSI = src2, EDX = stride1, ECX = stride2

    .text
    .intel_syntax noprefix

# ============================================================================
# 16x16 SAD using AVX-512
# ============================================================================
    .globl avx512_sad_16x16
    .type avx512_sad_16x16, @function
avx512_sad_16x16:
    # Arguments:
    # rdi = src1 pointer
    # rsi = src2 pointer
    # edx = stride1
    # ecx = stride2
    # Returns: eax = SAD value

    push rbp
    mov rbp, rsp

    # Zero accumulator
    vpxor zmm0, zmm0, zmm0     # SAD accumulator (32 x 16-bit)

    # Process 16 rows
    xor r8, r8                 # row counter

.L_sad16_loop:
    cmp r8, 16
    jge .L_sad16_done

    # Load 16 pixels from each source
    vmovdqu xmm1, [rdi]        # src1[y][0:15]
    vmovdqu xmm2, [rsi]        # src2[y][0:15]

    # Compute absolute differences using AVX-512
    vpmovzxbw ymm3, xmm1       # Expand to 16-bit
    vpmovzxbw ymm4, xmm2

    vpsubw ymm5, ymm3, ymm4    # Difference
    vpabsw ymm5, ymm5          # Absolute value

    # Accumulate into zmm0 (expand ymm5 to zmm)
    vpmovsxwd zmm6, ymm5       # Expand to 32-bit for accumulation
    vpaddd zmm0, zmm0, zmm6

    # Move to next row
    add rdi, rdx               # src1 += stride1
    add rsi, rcx               # src2 += stride2
    inc r8
    jmp .L_sad16_loop

.L_sad16_done:
    # Horizontal sum of zmm0
    vextracti64x4 ymm1, zmm0, 1
    vpaddd ymm0, ymm0, ymm1    # Add upper and lower halves

    vextracti128 xmm1, ymm0, 1
    vpaddd xmm0, xmm0, xmm1

    vpshufd xmm1, xmm0, 0x4E
    vpaddd xmm0, xmm0, xmm1

    vpshufd xmm1, xmm0, 0xB1
    vpaddd xmm0, xmm0, xmm1

    vmovd eax, xmm0            # Return SAD value

    pop rbp
    ret
    .size avx512_sad_16x16, .-avx512_sad_16x16

# ============================================================================
# 32x32 SAD using AVX-512
# ============================================================================
    .globl avx512_sad_32x32
    .type avx512_sad_32x32, @function
avx512_sad_32x32:
    push rbp
    mov rbp, rsp

    # Zero accumulator
    vpxor zmm0, zmm0, zmm0
    vpxor zmm1, zmm1, zmm1     # Second accumulator for 32 pixels

    xor r8, r8                 # row counter

.L_sad32_loop:
    cmp r8, 32
    jge .L_sad32_done

    # Load 32 pixels from each source (using YMM registers)
    vmovdqu ymm2, [rdi]        # src1[y][0:31]
    vmovdqu ymm3, [rsi]        # src2[y][0:31]

    # Process first 16 pixels
    vextracti128 xmm4, ymm2, 0
    vextracti128 xmm5, ymm3, 0
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpmovsxwd zmm7, ymm6
    vpaddd zmm0, zmm0, zmm7

    # Process second 16 pixels
    vextracti128 xmm4, ymm2, 1
    vextracti128 xmm5, ymm3, 1
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpmovsxwd zmm7, ymm6
    vpaddd zmm1, zmm1, zmm7

    add rdi, rdx
    add rsi, rcx
    inc r8
    jmp .L_sad32_loop

.L_sad32_done:
    # Combine accumulators
    vpaddd zmm0, zmm0, zmm1

    # Horizontal sum
    vextracti64x4 ymm1, zmm0, 1
    vpaddd ymm0, ymm0, ymm1

    vextracti128 xmm1, ymm0, 1
    vpaddd xmm0, xmm0, xmm1

    vpshufd xmm1, xmm0, 0x4E
    vpaddd xmm0, xmm0, xmm1

    vpshufd xmm1, xmm0, 0xB1
    vpaddd xmm0, xmm0, xmm1

    vmovd eax, xmm0

    pop rbp
    ret
    .size avx512_sad_32x32, .-avx512_sad_32x32

# ============================================================================
# 64x64 SAD using AVX-512
# ============================================================================
    .globl avx512_sad_64x64
    .type avx512_sad_64x64, @function
avx512_sad_64x64:
    push rbp
    mov rbp, rsp
    push rbx

    # Four accumulators for 64 pixels
    vpxor zmm0, zmm0, zmm0
    vpxor zmm1, zmm1, zmm1
    vpxor zmm2, zmm2, zmm2
    vpxor zmm3, zmm3, zmm3

    xor r8, r8                 # row counter

.L_sad64_loop:
    cmp r8, 64
    jge .L_sad64_done

    # Load 64 pixels in 4 chunks of 16
    # Chunk 0 (pixels 0-15)
    vmovdqu xmm4, [rdi]
    vmovdqu xmm5, [rsi]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpmovsxwd zmm7, ymm6
    vpaddd zmm0, zmm0, zmm7

    # Chunk 1 (pixels 16-31)
    vmovdqu xmm4, [rdi+16]
    vmovdqu xmm5, [rsi+16]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpmovsxwd zmm7, ymm6
    vpaddd zmm1, zmm1, zmm7

    # Chunk 2 (pixels 32-47)
    vmovdqu xmm4, [rdi+32]
    vmovdqu xmm5, [rsi+32]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpmovsxwd zmm7, ymm6
    vpaddd zmm2, zmm2, zmm7

    # Chunk 3 (pixels 48-63)
    vmovdqu xmm4, [rdi+48]
    vmovdqu xmm5, [rsi+48]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpmovsxwd zmm7, ymm6
    vpaddd zmm3, zmm3, zmm7

    add rdi, rdx
    add rsi, rcx
    inc r8
    jmp .L_sad64_loop

.L_sad64_done:
    # Combine all accumulators
    vpaddd zmm0, zmm0, zmm1
    vpaddd zmm2, zmm2, zmm3
    vpaddd zmm0, zmm0, zmm2

    # Horizontal sum
    vextracti64x4 ymm1, zmm0, 1
    vpaddd ymm0, ymm0, ymm1

    vextracti128 xmm1, ymm0, 1
    vpaddd xmm0, xmm0, xmm1

    vpshufd xmm1, xmm0, 0x4E
    vpaddd xmm0, xmm0, xmm1

    vpshufd xmm1, xmm0, 0xB1
    vpaddd xmm0, xmm0, xmm1

    vmovd eax, xmm0

    pop rbx
    pop rbp
    ret
    .size avx512_sad_64x64, .-avx512_sad_64x64

# ============================================================================
# AVX2 fallback implementations
# ============================================================================

    .globl avx2_sad_16x16
    .type avx2_sad_16x16, @function
avx2_sad_16x16:
    push rbp
    mov rbp, rsp

    vpxor ymm0, ymm0, ymm0
    xor r8, r8

.L_avx2_sad16_loop:
    cmp r8, 16
    jge .L_avx2_sad16_done

    vmovdqu xmm1, [rdi]
    vmovdqu xmm2, [rsi]

    vpmovzxbw ymm3, xmm1
    vpmovzxbw ymm4, xmm2
    vpsubw ymm5, ymm3, ymm4
    vpabsw ymm5, ymm5
    vpaddw ymm0, ymm0, ymm5

    add rdi, rdx
    add rsi, rcx
    inc r8
    jmp .L_avx2_sad16_loop

.L_avx2_sad16_done:
    # Horizontal sum
    vextracti128 xmm1, ymm0, 1
    vpaddw xmm0, xmm0, xmm1
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vmovd eax, xmm0
    and eax, 0xFFFF

    pop rbp
    ret
    .size avx2_sad_16x16, .-avx2_sad_16x16

    .globl avx2_sad_32x32
    .type avx2_sad_32x32, @function
avx2_sad_32x32:
    push rbp
    mov rbp, rsp

    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    xor r8, r8

.L_avx2_sad32_loop:
    cmp r8, 32
    jge .L_avx2_sad32_done

    vmovdqu ymm2, [rdi]
    vmovdqu ymm3, [rsi]

    vextracti128 xmm4, ymm2, 0
    vextracti128 xmm5, ymm3, 0
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpaddw ymm0, ymm0, ymm6

    vextracti128 xmm4, ymm2, 1
    vextracti128 xmm5, ymm3, 1
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpaddw ymm1, ymm1, ymm6

    add rdi, rdx
    add rsi, rcx
    inc r8
    jmp .L_avx2_sad32_loop

.L_avx2_sad32_done:
    vpaddw ymm0, ymm0, ymm1
    vextracti128 xmm1, ymm0, 1
    vpaddw xmm0, xmm0, xmm1
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vmovd eax, xmm0
    and eax, 0xFFFF

    pop rbp
    ret
    .size avx2_sad_32x32, .-avx2_sad_32x32

    .globl avx2_sad_64x64
    .type avx2_sad_64x64, @function
avx2_sad_64x64:
    push rbp
    mov rbp, rsp

    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    vpxor ymm2, ymm2, ymm2
    vpxor ymm3, ymm3, ymm3
    xor r8, r8

.L_avx2_sad64_loop:
    cmp r8, 64
    jge .L_avx2_sad64_done

    # Process 64 pixels in 4 chunks
    vmovdqu xmm4, [rdi]
    vmovdqu xmm5, [rsi]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpaddw ymm0, ymm0, ymm6

    vmovdqu xmm4, [rdi+16]
    vmovdqu xmm5, [rsi+16]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpaddw ymm1, ymm1, ymm6

    vmovdqu xmm4, [rdi+32]
    vmovdqu xmm5, [rsi+32]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpaddw ymm2, ymm2, ymm6

    vmovdqu xmm4, [rdi+48]
    vmovdqu xmm5, [rsi+48]
    vpmovzxbw ymm4, xmm4
    vpmovzxbw ymm5, xmm5
    vpsubw ymm6, ymm4, ymm5
    vpabsw ymm6, ymm6
    vpaddw ymm3, ymm3, ymm6

    add rdi, rdx
    add rsi, rcx
    inc r8
    jmp .L_avx2_sad64_loop

.L_avx2_sad64_done:
    vpaddw ymm0, ymm0, ymm1
    vpaddw ymm2, ymm2, ymm3
    vpaddw ymm0, ymm0, ymm2

    vextracti128 xmm1, ymm0, 1
    vpaddw xmm0, xmm0, xmm1
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vmovd eax, xmm0
    and eax, 0xFFFF

    pop rbp
    ret
    .size avx2_sad_64x64, .-avx2_sad_64x64

    .section .note.GNU-stack,"",@progbits
