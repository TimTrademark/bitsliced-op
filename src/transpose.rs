use core::arch::x86_64::*;

pub fn transpose_scalar(input: &[u64; 64]) -> [u64; 64] {
    let mut out = *input;
    let mut j = 32;
    let mut mask: u64 = 0x00000000FFFFFFFF;

    for _ in 0..6 {
        for k in 0..64 {
            if (k & j) == 0 {
                let x = out[k];
                let y = out[k | j];

                let t = (x ^ (y >> j)) & mask;
                out[k] = x ^ t;
                out[k | j] = y ^ (t << j);
            }
        }
        j >>= 1;
        mask ^= mask << j;
    }
    out
}

#[target_feature(enable = "avx512f,gfni,avx512vbmi,avx512bw")]
pub unsafe fn transpose_gfni(input: &[u64; 64]) -> [u64; 64] {
    let mut output = gather_bytes(input);
    //TODO: do i really need to go out of SIMD after gathering bytes?
    //i can keep data in 512 bit registers and avoid _mm512_set_epi64

    //next steps SIMD GFNI and interleave
    for index in 0..4 {
        let i = index * 16;

        let va = _mm512_set_epi64(
            output[i + 15] as i64,
            output[i + 13] as i64,
            output[i + 11] as i64,
            output[i + 9] as i64,
            output[i + 7] as i64,
            output[i + 5] as i64,
            output[i + 3] as i64,
            output[i + 1] as i64,
        );
        let vb = _mm512_set_epi64(
            output[i + 14] as i64,
            output[i + 12] as i64,
            output[i + 10] as i64,
            output[i + 8] as i64,
            output[i + 6] as i64,
            output[i + 4] as i64,
            output[i + 2] as i64,
            output[i + 0] as i64,
        );
        let gfni_va = gfni_bit_transpose_8x8(va);
        let gfni_vb = gfni_bit_transpose_8x8(vb);
        let interleaved = interleave_full(gfni_va, gfni_vb);
        output[i..i + 8].copy_from_slice(&interleaved.0);
        output[i + 8..i + 16].copy_from_slice(&interleaved.1);
    }

    output
}

unsafe fn gather_bytes(input: &[u64; 64]) -> [u64; 64] {
    let mut output = [[0u64; 8]; 8];
    let perm = make_perm();
    let idx = _mm512_loadu_si512(perm.as_ptr() as *const _);
    let ptr = input.as_ptr();
    for i in 0..8 {
        let v = _mm512_loadu_si512(ptr.add(i * 8) as *const _);

        let transposed = _mm512_permutexvar_epi8(idx, v);

        _mm512_storeu_si512(output[i].as_mut_ptr() as *mut _, transposed);
    }
    let mut aligned_output = [0u64; 64];
    for i in 0..8 {
        for j in 0..8 {
            aligned_output[i * 8 + j] = output[j][i];
        }
    }
    aligned_output
}

fn make_perm() -> [u8; 64] {
    let mut p = [0u8; 64];

    for byte_index in 0..8 {
        for lane in 0..8 {
            let dst = byte_index * 8 + lane;
            let src = lane * 8 + (7 - byte_index);
            p[dst] = src as u8;
        }
    }

    p
}

unsafe fn gfni_bit_transpose_8x8(x: __m512i) -> __m512i {
    // Each byte: bit i selects input bit i
    let matrix = _mm512_set1_epi64(0x8040201008040201u64 as i64);

    // Note the "qb" in the name:
    // q = matrix is treated as 64-bit (quadword)
    // b = input is treated as 8-bit (bytes)
    _mm512_gf2p8affine_epi64_epi8(matrix, x, 0)
}

//interleave bytes 0 and 1 from all rows (first step e.g interleave bytes)

unsafe fn interleave_full(va: __m512i, vb: __m512i) -> ([u64; 8], [u64; 8]) {
    /*
    let va = _mm512_set_epi64(
        h1 as i64, f1 as i64, d1 as i64, b1 as i64, h0 as i64, f0 as i64, d0 as i64, b0 as i64,
    );
    let vb = _mm512_set_epi64(
        g1 as i64, e1 as i64, c1 as i64, a1 as i64, g0 as i64, e0 as i64, c0 as i64, a0 as i64,
    );*/
    //(row1 comes here)...(row0) E00F00E01F01E02F02E03F03(hi) E04F04E05F05E06F06E07F07(lo) A00B00A01B01A02B02A03B03(hi) A04B04A05B05A06B06A07B07(lo)
    let lo = _mm512_unpacklo_epi8(va, vb);
    //...gh0hi gh0lo cd0hi cd0lo
    let hi = _mm512_unpackhi_epi8(va, vb);

    //...E04F04G04H04E05F05G05H05 E06F06G06H06E07F07G07H07 A04B04C04D04A05B05C05D05(hi) A06B06C06D06A07B07C07D07(lo)
    let lo_1 = _mm512_unpacklo_epi16(hi, lo);
    //...E00F00G00H00E01F01G01H01 E02F02G02H02E03F03G03H03 A00B00C00D00A01B01C01D01(hi) A02B02C02D02A03B03C03D03(lo)
    let hi_1 = _mm512_unpackhi_epi16(hi, lo);

    //shuffly lanes in right order
    //...(row1 before) (row0) A00B00C00D00A01B01C01D01(hi) A02B02C02D02A03B03C03D03(lo) A04B04C04D04A05B05C05D05(hi) A06B06C06D06A07B07C07D07(lo)
    let a = grab_128_lanes_a(lo_1, hi_1);
    //...(row1 before) (row0) E00F00G00H00E01F01G01H01 E02F02G02H02E03F03G03H03 E04F04G04H04E05F05G05H05 E06F06G06H06E07F07G07H07
    let b = grab_128_lanes_b(lo_1, hi_1);

    //...A02B02C02D02E02F02G02H02 A03B03C03D03E03F03G03H03 A06B06C06D06E06F06G06H06 A07B07C07D07E07F07G07H07
    let lo_2 = _mm512_unpacklo_epi32(b, a);
    //...A00B00C00D00E00F00G00H00 A01B01C01D01E01F01G01H01 A04B04C04D04E04F04G04H04 A05B05C05D05E05F05G05H05
    let hi_2 = _mm512_unpackhi_epi32(b, a);

    // row 0
    let l = _mm512_extracti64x2_epi64(hi_2, 1);
    let final_0 = _mm_extract_epi64(l, 1) as u64;
    let final_1 = _mm_cvtsi128_si64(l) as u64;

    let l = _mm512_extracti64x2_epi64(lo_2, 1);
    let final_2 = _mm_extract_epi64(l, 1) as u64;
    let final_3 = _mm_cvtsi128_si64(l) as u64;

    let l = _mm512_extracti64x2_epi64(hi_2, 0);
    let final_4 = _mm_extract_epi64(l, 1) as u64;
    let final_5 = _mm_cvtsi128_si64(l) as u64;

    let l = _mm512_extracti64x2_epi64(lo_2, 0);
    let final_6 = _mm_extract_epi64(l, 1) as u64;
    let final_7 = _mm_cvtsi128_si64(l) as u64;

    //row1
    let l = _mm512_extracti64x2_epi64(hi_2, 3);
    let final1_0 = _mm_extract_epi64(l, 1) as u64;
    let final1_1 = _mm_cvtsi128_si64(l) as u64;

    let l = _mm512_extracti64x2_epi64(lo_2, 3);
    let final1_2 = _mm_extract_epi64(l, 1) as u64;
    let final1_3 = _mm_cvtsi128_si64(l) as u64;

    let l = _mm512_extracti64x2_epi64(hi_2, 2);
    let final1_4 = _mm_extract_epi64(l, 1) as u64;
    let final1_5 = _mm_cvtsi128_si64(l) as u64;

    let l = _mm512_extracti64x2_epi64(lo_2, 2);
    let final1_6 = _mm_extract_epi64(l, 1) as u64;
    let final1_7 = _mm_cvtsi128_si64(l) as u64;

    let row0 = [
        final_0, final_1, final_2, final_3, final_4, final_5, final_6, final_7,
    ];
    let row1 = [
        final1_0, final1_1, final1_2, final1_3, final1_4, final1_5, final1_6, final1_7,
    ];
    (row0, row1)
}

#[target_feature(enable = "avx512f")]
unsafe fn grab_128_lanes_a(a: __m512i, b: __m512i) -> __m512i {
    let idx = _mm512_set_epi64(
        13, 12, // B2
        5, 4, // A2
        9, 8, // B0
        1, 0, // A0
    );

    _mm512_permutex2var_epi64(a, idx, b)
}

#[target_feature(enable = "avx512f")]
unsafe fn grab_128_lanes_b(a: __m512i, b: __m512i) -> __m512i {
    let idx = _mm512_set_epi64(
        15, 14, // B3
        7, 6, // A3
        11, 10, // B1
        3, 2, // A1
    );

    _mm512_permutex2var_epi64(a, idx, b)
}

unsafe fn print_m512i_bits(v: __m512i) {
    let mut buf = [0u64; 8];
    _mm512_storeu_si512(buf.as_mut_ptr() as *mut _, v);

    // Print highest bit first
    for &x in buf.iter().rev() {
        print!("{:064b}", x);
    }
    println!();
}

pub fn print_bit_matrix(matrix: &[u64]) {
    for (i, row) in matrix.iter().enumerate() {
        // Optional: print row index for debugging
        print!("{:2}: ", i);
        // Print the bits
        println!("{:064b}", row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn m512i_eq(a: __m512i, b: __m512i) -> bool {
        let mut aa = [0u64; 8];
        let mut bb = [0u64; 8];
        _mm512_storeu_si512(aa.as_mut_ptr() as *mut _, a);
        _mm512_storeu_si512(bb.as_mut_ptr() as *mut _, b);
        aa == bb
    }

    #[test]
    fn test_transpose_scalar() {
        let mut input = [0u64; 64];
        input[0] = 0xFFFFFFFFFFFFFFF0;
        input[1] = 0xFFFFFFFFFFFFFFF0;

        let transposed = transpose_scalar(&input);
        print_bit_matrix(&transposed);
        for i in 0..60 {
            assert_eq!(
                transposed[i],
                0b1100000000000000000000000000000000000000000000000000000000000000
            );
        }
        assert_eq!(transposed[60], 0);
        assert_eq!(transposed[61], 0);
        assert_eq!(transposed[62], 0);
        assert_eq!(transposed[63], 0);
    }

    #[test]
    fn test_gather_bytes() {
        let mut input = [0u64; 64];
        input[0] = 0xFFFFFFFFFFFFFFFF;
        input[1] = 0xFF;

        let mut output = [0u64; 64];
        //gather byte with index i for every 8 rows into a u64
        for i in 0..8 {
            let shift = (7 - i) * 8;
            for j in 0..8 {
                let mut acc = 0u64;
                for jj in 0..8 {
                    let row = input[j * 8 + jj];
                    let byte = ((row >> shift) as u8) as u64;
                    acc |= byte << (jj * 8);
                }
                output[i * 8 + j] = acc;
            }
        }
        print_bit_matrix(&output);

        unsafe {
            let output2 = gather_bytes(&input);
            print_bit_matrix(&output2);
            assert_eq!(output, output2);
        }
    }

    #[test]
    fn test_unpack() {
        let mut input = [0u64; 64];
        input[0] = 0xFFFFFFFFFFFFFFF0;
        input[1] = 0xFFFFFFFFFFFFFFF0;
        unsafe {
            let transposed = transpose_gfni(&input);
            print_bit_matrix(&transposed);
            for i in 0..60 {
                assert_eq!(
                    transposed[i],
                    0b1100000000000000000000000000000000000000000000000000000000000000
                );
            }
            assert_eq!(transposed[60], 0);
            assert_eq!(transposed[61], 0);
            assert_eq!(transposed[62], 0);
            assert_eq!(transposed[63], 0);
        }
    }

    //original data
    /*
    11111111...00000000
    11111111...00000000
    00000000...00000000
    ...
    00000000...00000000
    */
    //after GFNI
    /*
    byte 0 (MSB):
    11000000
    11000000
    11000000
    ...
    11000000
    in u64: 1100000011000000110000001100000011000000110000001100000011000000
    letter equivalent: A00A01A02...
    A00 (1 byte in length) = bit 0 of byte0 of rows 0..7, A01 = bit 1 of byte0 of rows 0..7, ...  A10  = bit 0 of byte 1 of rows 0..7
    B00 (1 byte in length) = bit 0 of byte0 of rows 8..15, B01 = bit 1 of byte0 of rows 8..15, ...  B10  = bit 0 of byte 1 of rows 8..15
    ...
    H00 (1 byte in length) = bit 0 of byte0 of rows 56..63, H01 = bit 1 of byte0 of rows 56..63, ...  H10  = bit 0 of byte 1 of rows 56..63
    Input:
    A00A01A02A03A04A05A06A07 (64bits length for every row)
    B00B01B02...
    ...
    A10A11A12A13A14A15A16A17
    ...
    H70H71H72H73H74H75H76H77
    Goal:
    A00B00...H00 (64bits length for every row)
    A01B01...H01
    ...
    Interleave bytes
    A00B00A01B01A02B02A03B03
    A04B04A05B05A06B06A07B07
    C00D00C01D01C02D02C03D03
    C04D04C05D05C06D06C07D07

    A00B00C00D00A01B01C01D01
    A02B02C02D02A03B03C03D03
    A04B04C04D04A05B05C05D05
    A06B06C06D06A07B07C07D07
    */
    #[test]
    fn test_gfni() {
        let mut gfni_input = [0u64; 8];
        gfni_input[0] = 0xFF00000000000000;
        unsafe {
            let input = _mm512_set_epi64(
                gfni_input[7] as i64,
                gfni_input[6] as i64,
                gfni_input[5] as i64,
                gfni_input[4] as i64,
                gfni_input[3] as i64,
                gfni_input[2] as i64,
                gfni_input[1] as i64,
                gfni_input[0] as i64,
            );
            let output = gfni_bit_transpose_8x8(input);
            let expected = _mm512_set_epi64(
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0b0000000100000001000000010000000100000001000000010000000100000001 as u64 as i64,
            );
            assert_eq!(m512i_eq(output, expected), true);
        }
    }

    #[test]
    fn test_interleave() {
        let a: u64 = 0b1100000011000000110000001100000011000000110000001100000011000000;
        let expected = 0b1100000000000000000000000000000000000000000000000000000000000000;

        unsafe {
            let va = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
            let vb = _mm512_set_epi64(0, 0, 0, a as i64, 0, 0, 0, a as i64);
            let interleaved = interleave_full(va, vb);
            for i in interleaved.0 {
                assert_eq!(expected, i);
            }
            for i in interleaved.1 {
                assert_eq!(expected, i);
            }
        }
    }

    #[test]
    fn test_grab_128_lanes() {
        let ones: u64 = 0xFFFFFFFFFFFFFFFF;
        let unique: u64 = 0xFF;
        unsafe {
            let a = _mm512_set_epi64(
                0,
                0,
                ones as i64,
                ones as i64,
                0,
                0,
                ones as i64,
                ones as i64,
            );
            let b = _mm512_set_epi64(
                0,
                0,
                unique as i64,
                unique as i64,
                0,
                0,
                unique as i64,
                unique as i64,
            );
            let output = grab_128_lanes_a(a, b);
            let expected = _mm512_set_epi64(
                unique as i64,
                unique as i64,
                ones as i64,
                ones as i64,
                unique as i64,
                unique as i64,
                ones as i64,
                ones as i64,
            );
            assert_eq!(m512i_eq(output, expected), true);
            let a = _mm512_set_epi64(
                ones as i64,
                ones as i64,
                0,
                0,
                ones as i64,
                ones as i64,
                0,
                0,
            );
            let b = _mm512_set_epi64(
                unique as i64,
                unique as i64,
                0,
                0,
                unique as i64,
                unique as i64,
                0,
                0,
            );
            let output = grab_128_lanes_b(a, b);
            let expected = _mm512_set_epi64(
                unique as i64,
                unique as i64,
                ones as i64,
                ones as i64,
                unique as i64,
                unique as i64,
                ones as i64,
                ones as i64,
            );
            assert_eq!(m512i_eq(output, expected), true);
        }
    }
}
