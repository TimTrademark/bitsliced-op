use core::arch::x86_64::*;

//interleave bytes 0 and 1 from rows 0-7 (first step e.g interleave bytes)
#[target_feature(enable = "avx512bw")]
unsafe fn interleave_u64_bytes(
    a0: u64,
    b0: u64,
    c0: u64,
    d0: u64,
    e0: u64,
    f0: u64,
    g0: u64,
    h0: u64,
    a1: u64,
    b1: u64,
    c1: u64,
    d1: u64,
    e1: u64,
    f1: u64,
    g1: u64,
    h1: u64,
) -> (
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
) {
    let va = _mm512_set_epi64(
        h1 as i64, h0 as i64, f1 as i64, f0 as i64, d1 as i64, d0 as i64, b1 as i64, b0 as i64,
    );
    let vb = _mm512_set_epi64(
        g1 as i64, g0 as i64, e1 as i64, e0 as i64, c1 as i64, c0 as i64, a1 as i64, a0 as i64,
    );

    let lo = _mm512_unpacklo_epi8(va, vb);
    let hi = _mm512_unpackhi_epi8(va, vb);
    print_m512i_bits(lo);
    print_m512i_bits(hi);

    // row 0
    let l = _mm512_extracti64x2_epi64(lo, 0);
    let ab0_lo = _mm_cvtsi128_si64(l) as u64;
    let ab0_hi = _mm_extract_epi64(l, 1) as u64;

    let l = _mm512_extracti64x2_epi64(lo, 1);
    let cd0_lo = _mm_cvtsi128_si64(l) as u64;
    let cd0_hi = _mm_extract_epi64(l, 1) as u64;

    let l = _mm512_extracti64x2_epi64(lo, 2);
    let ef0_lo = _mm_cvtsi128_si64(l) as u64;
    let ef0_hi = _mm_extract_epi64(l, 1) as u64;

    let l = _mm512_extracti64x2_epi64(lo, 3);
    let gh0_lo = _mm_cvtsi128_si64(l) as u64;
    let gh0_hi = _mm_extract_epi64(l, 1) as u64;

    //row 1
    let l = _mm512_extracti64x2_epi64(hi, 0);
    let ab1_lo = _mm_cvtsi128_si64(l) as u64;
    let ab1_hi = _mm_extract_epi64(l, 1) as u64;

    let l = _mm512_extracti64x2_epi64(hi, 1);
    let cd1_lo = _mm_cvtsi128_si64(l) as u64;
    let cd1_hi = _mm_extract_epi64(l, 1) as u64;

    let l = _mm512_extracti64x2_epi64(hi, 2);
    let ef1_lo = _mm_cvtsi128_si64(l) as u64;
    let ef1_hi = _mm_extract_epi64(l, 1) as u64;

    let l = _mm512_extracti64x2_epi64(hi, 3);
    let gh1_lo = _mm_cvtsi128_si64(l) as u64;
    let gh1_hi = _mm_extract_epi64(l, 1) as u64;

    (
        ab0_hi, ab0_lo, cd0_hi, cd0_lo, ef0_hi, ef0_lo, gh0_hi, gh0_lo, ab1_hi, ab1_lo, cd1_hi,
        cd1_lo, ef1_hi, ef1_lo, gh1_hi, gh1_lo,
    )
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
    fn test_gfni_unpack_transpose() {
        let mut gfni_input = [0u64; 64];
        gfni_input[0] = 0b1100000011000000110000001100000011000000110000001100000011000000;
        gfni_input[1] = 0b1100000011000000110000001100000011000000110000001100000011000000;
        print_bit_matrix(&gfni_input);
    }

    #[test]
    fn test_interleave() {
        let a: u64 = 0xFF000000000000FF;
        let b: u64 = 0;
        unsafe {
            let out = interleave_u64_bytes(a, b, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0);
            println!("{:064b}", out.0);
            println!("{:064b}", out.1);
            assert_eq!(
                out.0,
                0b1111111100000000000000000000000000000000000000000000000000000000
            );
            assert_eq!(
                out.1,
                0b0000000000000000000000000000000000000000000000001111111100000000
            );
            assert_eq!(
                out.8,
                0b1111111100000000000000000000000000000000000000000000000000000000
            );
            assert_eq!(
                out.9,
                0b0000000000000000000000000000000000000000000000001111111100000000
            );
        }
    }
}
