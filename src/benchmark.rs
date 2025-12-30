use std::time::{Duration, Instant};

struct BenchmarkResult {
    runs: u64,
    duration: Duration,
}

pub fn benchmark<F>(name: &str, runs: u64, warmup: u64, parallel_count: u64, func: F)
where
    F: Fn(),
{
    for _ in 0..warmup {
        func();
    }
    let start = Instant::now();
    for _ in 0..runs {
        func();
    }
    let duration = start.elapsed();
    let result = BenchmarkResult { runs, duration };
    let operations_per_second =
        (result.runs as f64 / result.duration.as_secs_f64()) * parallel_count as f64;
    print_report(name, operations_per_second);
}

fn print_report(name: &str, hashes_per_second: f64) {
    println!(
        "{} ran at {}",
        name,
        format_operations_per_second(hashes_per_second)
    );
}

fn format_operations_per_second(hashes_per_second: f64) -> String {
    let gh_delimiter = 1_000_000_000.0;
    let mh_delimiter = 1_000_000.0;
    let kh_delimiter = 1_000.0;
    if hashes_per_second > gh_delimiter {
        format!("{:.2}GH/s", hashes_per_second / gh_delimiter)
    } else if hashes_per_second > mh_delimiter {
        format!("{:.2}MH/s", hashes_per_second / mh_delimiter)
    } else if hashes_per_second > kh_delimiter {
        format!("{:.2}KH/s", hashes_per_second / kh_delimiter)
    } else {
        format!("{:.2}H/s", hashes_per_second)
    }
}
