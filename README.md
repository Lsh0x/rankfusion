# rankfusion

[![CI](https://github.com/Lsh0x/rankfusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Lsh0x/rankfusion/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/rankfusion.svg)](https://crates.io/crates/rankfusion)
[![docs.rs](https://docs.rs/rankfusion/badge.svg)](https://docs.rs/rankfusion)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Backend-agnostic rank aggregation, score fusion, and reranking for Rust.**

```toml
[dependencies]
rankfusion = "0.1"
```

## What it is

`rankfusion` is **not** a search engine. It does not perform retrieval and knows
nothing about ANN indexes, BM25, embeddings, databases, documents, or storage.

Its single responsibility: **given multiple ranked candidate lists produced by
external systems, combine them into one optimized ranking** using configurable
fusion and reranking strategies.

```text
External sources
      │
 ┌────┼────┐
 │    │    │
List A  List B  List C
 │    │    │
 └────┼────┘
      ▼
Rank aggregation (RRF, weighted RRF, linear fusion)
      ▼
Reranking pipeline (composable, in-place stages)
      ▼
Final ranked results
```

It is a generic ranking layer usable by search engines, recommendation systems,
RAG pipelines, personalization systems, data discovery — anything that produces
ranked candidate lists. Retrieval and ranking are separate concerns; this crate
is only the second one.

## Quick start

Two retrieval systems answered the same query. Their scores are not
comparable — only their order is. RRF turns that into one ranking:

```rust
use rankfusion::{Pipeline, RankedList, Rrf, Scored, TopK};

let vector_search: RankedList<&str> = ["a", "b", "c"].into_iter().collect();
let keyword_search: RankedList<&str> = ["b", "d", "a"].into_iter().collect();

let freshness_boost = |candidates: &mut Vec<Scored<&str, ()>>| {
    for c in candidates.iter_mut() {
        if *c.id() == "d" {
            c.score *= 1.5;
        }
    }
    candidates.sort_by(Scored::cmp_score_desc);
};

let results = Pipeline::new(Rrf::default())
    .reranker(freshness_boost)
    .reranker(TopK::new(3))
    .rank(vec![vector_search, keyword_search])
    .unwrap();

assert_eq!(*results[0].id(), "b"); // ranked highly by both sources
```

Runnable examples live in [`examples/`](examples): `cargo run --example
basic_rrf`, `--example explain`, `--example pipeline`.

## Features

- **Core data model** — `Candidate<Id, Metadata>` with generic identifiers
  (`u64`, UUID, `String`, custom) and opaque metadata; explicit merge policy for
  candidates appearing in multiple lists.
- **Fusion strategies** — Reciprocal Rank Fusion (RRF), weighted RRF, linear
  score fusion. `O(total_candidates + n log n)`, hash-map accumulation, no
  quadratic scans.
- **Score normalization** — min-max, z-score, softmax, with explicit edge-case
  policies (single-element lists, NaN, ±inf).
- **Reranking pipeline** — composable, synchronous, in-place `Reranker` stages
  (freshness boost, business rules, custom scoring). Async rerankers adapt
  outside the core.
- **Explainability** — per-source score contribution tracing via
  `fuse_explained`, off the hot path.

### Optional feature flags

| Feature | Effect |
|---------|--------|
| `ahash` | Faster hashing for the fusion accumulators; `std::collections::HashMap` by default. |
| `eval`  | Ranking-quality metrics: `ndcg@k`, MRR, `recall@k`. Zero extra dependencies. |
| `serde` | `Serialize`/`Deserialize` on the core data model and explainability types. |

## Evaluating your fusion config

The optional `eval` feature ships `ndcg@k`, MRR and `recall@k` to compare
fusion configurations against your own ground truth — RRF `k=60` vs `k=20`,
RRF vs linear fusion, weight tuning:

```toml
rankfusion = { version = "0.1", features = ["eval"] }
```

```rust
use rankfusion::eval::ndcg_at_k;
// fuse with each candidate config, then score the rankings against judgments
let quality = ndcg_at_k(&ranking_ids, &judgments, 10);
```

Criterion benchmarks live in `benches/` (`cargo bench`) and back the
`O(total + n log n)` complexity claim.

## Non-goals

ANN indexes, vector databases, BM25, tokenizers, embeddings, document storage,
query parsing. Those belong to the external systems feeding this library.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the design invariants and the checks
CI enforces. Vulnerabilities: [SECURITY.md](SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
