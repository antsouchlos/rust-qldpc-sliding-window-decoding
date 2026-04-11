use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, black_box};
use rust_qldpc::bp::{Settings, SyndromeBpDecoder};
use rust_qldpc::bp_core::min_sum::SyndromeMinSumCore;
use rust_qldpc::bp_core::spa::SyndromeSpaCore;
use rust_qldpc::bp_core::SyndromeBpStrategy;
use rust_qldpc::Decoder;
use sprs::TriMat;

/// Build a (3,6)-regular LDPC parity-check matrix of size `n` × `2n`.
///
/// Uses 6 circulant permutation matrices (each n×n) arranged as 3 in each
/// column block. Each row has degree 6, each column has degree 3.
fn make_ldpc(n: usize) -> sprs::CsMat<u8> {
    let mut m = TriMat::<u8>::new((n, 2 * n));

    // Column block 0 (cols 0..n): shifts chosen to avoid trivial cycles
    let shifts0 = [0usize, n / 7 + 1, 2 * n / 7 + 1];
    for shift in shifts0 {
        for i in 0..n {
            m.add_triplet(i, (i + shift) % n, 1);
        }
    }

    // Column block 1 (cols n..2n)
    let shifts1 = [0usize, n / 5 + 1, 3 * n / 7 + 1];
    for shift in shifts1 {
        for i in 0..n {
            m.add_triplet(i, n + (i + shift) % n, 1);
        }
    }

    m.to_csr()
}

fn bench_spa(c: &mut Criterion) {
    let n = 100;
    let h = make_ldpc(n);
    let channel_llrs = vec![2.0f64; h.cols()];
    let syndrome = vec![0u8; h.rows()];

    let mut group = c.benchmark_group("spa");

    // Benchmark vn_update and cn_update in isolation with pre-seeded messages.
    {
        let mut core = SyndromeSpaCore::new(&h, &channel_llrs);
        for edge in &mut core.state.edges {
            edge.msg_cn_to_vn = 0.5;
            edge.msg_vn_to_cn = 0.5;
        }

        group.bench_function("vn_update", |b| {
            b.iter(|| core.vn_update());
        });

        group.bench_function("cn_update", |b| {
            b.iter(|| core.cn_update(black_box(&syndrome)));
        });
    }

    // Full decode benchmarked using iter_batched so each sample starts with a
    // fresh decoder. The syndrome has one bit set so the decoder doesn't
    // converge and runs all max_iter=30 iterations.
    {
        let mut syndrome_hard = syndrome.clone();
        syndrome_hard[0] = 1;

        group.bench_function("decode/30iter", |b| {
            b.iter_batched(
                || {
                    SyndromeBpDecoder::<SyndromeSpaCore>::new(
                        Settings { max_iter: 30 },
                        &h,
                        &channel_llrs,
                    )
                },
                |mut decoder| decoder.decode(black_box(&syndrome_hard)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_min_sum(c: &mut Criterion) {
    let n = 100;
    let h = make_ldpc(n);
    let channel_llrs = vec![2.0f64; h.cols()];
    let syndrome = vec![0u8; h.rows()];

    let mut group = c.benchmark_group("min_sum");

    {
        let mut core = SyndromeMinSumCore::new(&h, &channel_llrs);
        for edge in &mut core.0.edges {
            edge.msg_cn_to_vn = 0.5;
            edge.msg_vn_to_cn = 0.5;
        }

        group.bench_function("vn_update", |b| {
            b.iter(|| core.vn_update());
        });

        group.bench_function("cn_update", |b| {
            b.iter(|| core.cn_update(black_box(&syndrome)));
        });
    }

    {
        let mut syndrome_hard = syndrome.clone();
        syndrome_hard[0] = 1;

        group.bench_function("decode/30iter", |b| {
            b.iter_batched(
                || {
                    SyndromeBpDecoder::<SyndromeMinSumCore>::new(
                        Settings { max_iter: 30 },
                        &h,
                        &channel_llrs,
                    )
                },
                |mut decoder| decoder.decode(black_box(&syndrome_hard)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Sweep over matrix sizes to see how cn_update throughput scales.
fn bench_matrix_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("spa/cn_update_vs_size");

    for &n in &[50usize, 100, 200, 500] {
        let h = make_ldpc(n);
        let channel_llrs = vec![2.0f64; h.cols()];
        let syndrome = vec![0u8; h.rows()];

        let mut core = SyndromeSpaCore::new(&h, &channel_llrs);
        for edge in &mut core.state.edges {
            edge.msg_vn_to_cn = 0.5;
        }

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| core.cn_update(black_box(&syndrome)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_spa, bench_min_sum, bench_matrix_size);
criterion_main!(benches);
