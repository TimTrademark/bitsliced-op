use std::{env, hint::black_box};

use bitsliced_op::{
    benchmark::benchmark,
    bitsliced_add, bitsliced_add_inline, bitsliced_modulo_power_of_two,
    bitsliced_modulo_power_of_two_inline,
    transpose::{transpose_gfni, transpose_scalar},
    transpose_64x64,
};
use wide::u64x8;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: bitsliced-op <benchmark_name>");
    }
    let benchmark_name = &args[1];
    start_benchmark(benchmark_name.as_str());
}

fn start_benchmark(benchmark_name: &str) {
    match benchmark_name {
        "na" | "normal_addition" => {
            benchmark("normal_addition", 1_000_000, 10000, 1, || {
                let result = black_box(1) + black_box(1);
                black_box(result);
            });
        }
        "ba" | "bitsliced_addition" => {
            let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
            let zero = u64x8::ZERO;
            let mut a = [zero; 64];
            a[63] = all_ones;
            let mut b = [zero; 64];
            b[63] = all_ones;

            benchmark("bitsliced_addition", 1_000_000, 10000, 64, || {
                let _ = bitsliced_add(&a, &b);
            });
        }
        "bai" | "bitsliced_addition_inline" => {
            let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
            let zero = u64x8::ZERO;
            let mut a = [zero; 64];
            a[63] = all_ones;
            let mut b = [zero; 64];
            b[63] = all_ones;

            benchmark("bitsliced_addition_inline", 1_000_000, 10000, 64, || {
                bitsliced_add_inline(&mut a, &b);
            });
        }
        "bm" | "bitsliced_modulo" => {
            let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
            let a = [all_ones; 64];

            benchmark("bitsliced_modulo", 1_000_000, 10000, 64, || {
                let _ = bitsliced_modulo_power_of_two(&a, 56);
            });
        }
        "bmi" | "bitsliced_modulo_inline" => {
            let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
            let mut a = [all_ones; 64];

            benchmark("bitsliced_modulo_inline", 1_000_000, 10000, 64, || {
                let _ = bitsliced_modulo_power_of_two_inline(&mut a, 56);
            });
        }
        "ts" | "transpose_scalar" => {
            let transpose_input = [0u64; 64];

            benchmark("transpose_scalar", 1_000_000, 10000, 1, || {
                let _ = transpose_scalar(&transpose_input);
            });
        }
        "tg" | "transpose_gfni" => {
            let transpose_input = [0u64; 64];

            benchmark("transpose_gfni", 1_000_000, 10000, 1, || unsafe {
                let _ = transpose_gfni(&transpose_input);
            });
        }
        "tr" | "transpose" => {
            let transpose_input = [0u64; 64];

            benchmark("transpose", 1_000_000, 10000, 1, || {
                let _ = transpose_64x64(&transpose_input);
            });
        }
        _ => {
            println!("Invalid benchmark name")
        }
    }
}
