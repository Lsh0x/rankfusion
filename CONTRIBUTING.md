# Contributing to rankfusion

Thanks for taking the time. Bug reports, benchmarks that contradict the
complexity claims, and new fusion strategies backed by a reference are all
welcome.

## Scope

`rankfusion` is a ranking layer, not a search engine. Retrieval — ANN indexes,
BM25, tokenizers, embeddings, storage, query parsing — is explicitly out of
scope and belongs to the systems feeding this library. A change that pulls any
of it in will be declined regardless of quality.

## Before opening a pull request

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

All four run in CI, plus an MSRV check (Rust 1.70), `cargo-deny`, and coverage.

## What a change is expected to carry

- **Tests.** Unit tests live next to the code in `#[cfg(test)] mod tests`;
  cross-module and public-API tests go in `tests/`. Anything with a
  mathematical invariant (ordering, score sums, normalization bounds) belongs
  in `tests/properties.rs` as a `proptest`, not only as hand-picked cases.
- **Docs.** `#![warn(missing_docs)]` is enforced: every public item needs a
  doc comment. Non-obvious behaviour — degenerate inputs, NaN handling, error
  cases — is documented at the module level with the reasoning, not just the
  what.
- **Benchmarks**, when touching a hot path. `cargo bench` runs the Criterion
  suite in `benches/`.

## Design invariants

These hold crate-wide; a change that breaks one needs to argue the case first,
in an issue:

- **No retrieval.** See Scope above.
- **Score status is carried by the type.** `RankedList` for rank-based input,
  `ScoredList` for score-based input — never `Option<f32>` on a candidate.
- **NaN is garbage-in/garbage-out, never a panic or a `Result`.** All score
  comparisons go through `f32::total_cmp`, so ordering stays total and
  deterministic. Degenerate cases in normalization are *defined* (see
  `src/normalization/mod.rs`), never left to produce NaN.
- **Explainability is off the hot path.** `fuse_explained` uses a separate
  accumulation path; `fuse` pays nothing for its existence.
- **Feature flags are additive.** `default = []`; no feature may change
  behaviour, only add API surface or swap an implementation detail (`ahash`).

## Commits and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org)
(`feat:`, `fix:`, `docs:`, `perf:`, `chore:`) — releases and the changelog are
generated from them by [release-plz](https://release-plz.dev), so the prefix
determines the next version. Breaking changes must say so in the footer
(`BREAKING CHANGE: …`).

## Licence

Contributions are dual-licensed under MIT and Apache-2.0, matching the crate.
By submitting a pull request you agree to license your work under those terms.
