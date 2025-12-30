use wide::u64x8;

pub mod benchmark;
//expects the input to be in bitsliced form e.g integers are columns, not rows
//last row is LSB
pub fn bitsliced_add(a: &[u64x8; 64], b: &[u64x8; 64]) -> [u64x8; 64] {
    let mut carry = u64x8::ZERO;
    let mut sum = [u64x8::ZERO; 64];
    for i in (0..64).rev() {
        let res = calc_sum_carry(a[i], b[i], carry);
        sum[i] = res.0;
        //only set carry if we haven't reached the end yet, we currently ignore overflows
        carry = res.1;
    }
    sum
}

fn calc_sum_carry(a: u64x8, b: u64x8, carry: u64x8) -> (u64x8, u64x8) {
    let sum = a ^ b ^ carry;
    let next_carry = (a & b) | (carry & (a ^ b));
    (sum, next_carry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_works() {
        let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
        let zero = u64x8::ZERO;
        let mut a = [zero; 64];
        a[63] = all_ones;
        let mut b = [zero; 64];
        b[63] = all_ones;
        let sum = bitsliced_add(&a, &b);
        assert_eq!(sum[63], zero);
        assert_eq!(sum[62], all_ones);
        assert_eq!(sum[61], zero);
    }
}
