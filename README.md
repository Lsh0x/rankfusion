# rankfusion

[![crates.io](https://img.shields.io/crates/v/rankfusion.svg)](https://crates.io/crates/rankfusion)
[![docs.rs](https://docs.rs/rankfusion/badge.svg)](https://docs.rs/rankfusion)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Backend-agnostic rank aggregation, score fusion, and reranking for Rust.**

> ⚠️ **Status: early design phase (`0.0.x`).** The API is being designed in the
> open — expect breaking changes until `0.1.0`. Follow the
> [roadmap issues](https://github.com/Lsh0x/rankfusion/issues) to see where it's going.

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

## Planned for 0.1.0

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
- **Explainability** — optional per-source score contribution tracing.
- **`ahash` feature flag** — optional faster hashing; `std::collections::HashMap`
  by default.

## Non-goals

ANN indexes, vector databases, BM25, tokenizers, embeddings, document storage,
query parsing. Those belong to the external systems feeding this library.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
