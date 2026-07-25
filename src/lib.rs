//! # rankfusion
//!
//! Backend-agnostic rank aggregation, score fusion, and reranking.
//!
//! This library is **not** a search engine. It does not perform retrieval and
//! knows nothing about ANN, BM25, embeddings, databases, or storage. Its only
//! responsibility: given multiple ranked candidate lists produced by external
//! systems, combine them into a single optimized ranking using configurable
//! fusion and reranking strategies.
//!
//! ```text
//! External sources ──▶ RankedList / ScoredList ──▶ fusion ──▶ reranking ──▶ Vec<Scored>
//! ```
//!
//! ## Data model
//!
//! Rank-based inputs and score-based inputs are distinct types — see the
//! [`core`] module docs for the full design rationale (score
//! status, metadata merge policy, NaN policy):
//!
//! - [`RankedList`]: ordered candidates, rank implicit — feeds rank fusion (RRF).
//! - [`ScoredList`]: ordered + uniformly scored — feeds score fusion; converts
//!   to [`RankedList`] for free.
//! - Fusion always outputs [`Scored`] candidates: a fused score always exists.
//!
//! ## Status
//!
//! `0.0.x` — API under active design, expect breaking changes until `0.1.0`.
//! The roadmap lives in the [repository issues](https://github.com/Lsh0x/rankfusion/issues).

pub mod core;
pub mod fusion;

pub use crate::core::{Candidate, FirstWins, MergePolicy, RankedList, Scored, ScoredList};
pub use crate::fusion::{FusionError, Rrf, WeightedRrf};
