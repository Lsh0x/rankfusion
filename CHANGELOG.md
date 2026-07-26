# Changelog

All notable changes to this project will be documented in this file.
Maintained by [release-plz](https://release-plz.dev/) from
[conventional commits](https://www.conventionalcommits.org/).

## [0.2.0](https://github.com/Lsh0x/rankfusion/compare/v0.1.0...v0.2.0) - 2026-07-26

### Changed

- **MSRV is now 1.71** (was 1.70). The optional `serde` feature pulls
  `serde_derive`, whose proc-macro dependencies raised their own MSRV.
  Released as a minor bump rather than a patch so that a `0.1` requirement
  does not pick this up automatically and break builds on 1.70.

### Added

- optional serde feature, property tests, and prod-ready hardening ([#19](https://github.com/Lsh0x/rankfusion/pull/19))

### Other

- bump actions/checkout to v7 and codecov/codecov-action to v7 ([#27](https://github.com/Lsh0x/rankfusion/pull/27))
- *(deps)* bump criterion from 0.5.1 to 0.8.2 ([#23](https://github.com/Lsh0x/rankfusion/pull/23))
- compile benchmarks in CI, exempt the MSRV toolchain pin from Dependabot ([#25](https://github.com/Lsh0x/rankfusion/pull/25))

## [0.1.0] - 2026-07-25

First real release — the complete v0.1.0 milestone.

### Added

- Core data model: `Candidate`, `Scored`, `RankedList` (rank-based inputs) and
  `ScoredList` (score-based inputs), free one-way `ScoredList → RankedList`
  conversion, `MergePolicy` with `FirstWins` default, deterministic NaN policy
  via `f32::total_cmp` (#9)
- Reciprocal Rank Fusion (`Rrf`, default `k = 60`) and `WeightedRrf` —
  single-pass hash-map accumulation, `O(total + n log n)`, deterministic
  first-seen tie-breaking, typed `FusionError::WeightCountMismatch` (#11)
- Linear score fusion (`LinearFusion`) with per-source-list normalization:
  `MinMax`, `ZScore`, `Softmax`, explicit edge-case policies (#12)
- `Fusion` trait (input type enforced per strategy), synchronous in-place
  `Reranker` trait (closures welcome), composable `Pipeline`, `TopK` stage (#13)
- Explainability: `rank_explained` / `fuse_explained` return per-source
  contributions (`list_index`, `rank`, `partial_score`) that sum to the fused
  score; zero hot-path overhead (#14)
- Optional `ahash` feature — faster accumulator maps, zero default
  dependencies preserved (#15)
- Optional `eval` feature — `ndcg_at_k`, `reciprocal_rank`/`mrr`,
  `recall_at_k`; criterion benchmarks (#16)
- CI: fmt, clippy `-D warnings`, 4-way feature test matrix, rustdoc
  `-D warnings`, MSRV 1.70 (#10)

## [0.0.1] - 2026-07-25

Name reservation placeholder.
