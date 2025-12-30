use std::env;

use bitsliced_op::{benchmark::benchmark, bitsliced_add};
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
        "n" | "normal" => {
            benchmark("normal_addition", 100_000_000, 10000, 1, || {
                let _: u64 = 1 + 1;
            });
        }
        "b" | "bitsliced" => {
            let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
            let zero = u64x8::ZERO;
            let mut a = [zero; 64];
            a[63] = all_ones;
            let mut b = [zero; 64];
            b[63] = all_ones;

            benchmark("bitsliced_addition", 100_000_000, 10000, 64, || {
                let _ = bitsliced_add(&a, &b);
            });
        }
        _ => {
            println!("Invalid benchmark name")
        }
    }
}
