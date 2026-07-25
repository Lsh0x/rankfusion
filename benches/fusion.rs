//! Fusion benchmarks — defend the `O(total + n log n)` claim.
//!
//! Shapes: many small lists (10 × 1k) and few large ones (3 × 100k).
//! Comparing `rrf_3x100k` against `rrf_10x1k` (30× the candidates) is the
//! linear-scaling check: wall time should grow roughly with total candidate
//! count, not quadratically.

use criterion::{criterion_group, criterion_main, Criterion};
use rankfusion::{LinearFusion, RankedList, Rrf, Scored, ScoredList};
use std::hint::black_box;

/// `n_lists` lists of `len` candidates each, ~50% overlap between
/// neighbouring lists.
fn ranked_lists(n_lists: u64, len: u64) -> Vec<RankedList<u64>> {
    (0..n_lists)
        .map(|i| (i * len / 2..i * len / 2 + len).collect())
        .collect()
}

fn scored_lists(n_lists: u64, len: u64) -> Vec<ScoredList<u64>> {
    (0..n_lists)
        .map(|i| {
            ScoredList::new(
                (i * len / 2..i * len / 2 + len)
                    .map(|id| Scored::new(id, 1.0 / (id % 97 + 1) as f32, ()))
                    .collect(),
            )
        })
        .collect()
}

fn bench_fusion(c: &mut Criterion) {
    c.bench_function("rrf_10x1k", |b| {
        let lists = ranked_lists(10, 1_000);
        b.iter(|| Rrf::default().fuse(black_box(lists.clone())));
    });

    c.bench_function("rrf_3x100k", |b| {
        let lists = ranked_lists(3, 100_000);
        b.iter(|| Rrf::default().fuse(black_box(lists.clone())));
    });

    c.bench_function("linear_3x100k", |b| {
        let lists = scored_lists(3, 100_000);
        b.iter(|| {
            LinearFusion::new(vec![1.0, 1.0, 1.0])
                .fuse(black_box(lists.clone()))
                .unwrap()
        });
    });

    c.bench_function("rrf_explained_10x1k", |b| {
        let lists = ranked_lists(10, 1_000);
        b.iter(|| Rrf::default().fuse_explained(black_box(lists.clone())));
    });
}

criterion_group!(benches, bench_fusion);
criterion_main!(benches);
