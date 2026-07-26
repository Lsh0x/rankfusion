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
//! ## Example
//!
//! ```
//! use rankfusion::{Pipeline, RankedList, Rrf, Scored, TopK};
//!
//! // ranked lists from two external systems (vector search, keyword search…)
//! let list_a: RankedList<&str> = ["a", "b", "c"].into_iter().collect();
//! let list_b: RankedList<&str> = ["b", "d", "a"].into_iter().collect();
//!
//! let freshness_boost = |candidates: &mut Vec<Scored<&str, ()>>| {
//!     for c in candidates.iter_mut() {
//!         if *c.id() == "d" {
//!             c.score *= 1.5;
//!         }
//!     }
//!     candidates.sort_by(Scored::cmp_score_desc);
//! };
//!
//! let results = Pipeline::new(Rrf::default())
//!     .reranker(freshness_boost)
//!     .reranker(TopK::new(3))
//!     .rank(vec![list_a, list_b])
//!     .unwrap();
//!
//! assert_eq!(results.len(), 3);
//! assert_eq!(*results[0].id(), "b");
//! ```
//!
//! ## Feature flags
//!
//! - `ahash` — faster hashing for the fusion accumulators.
//! - `eval` — ranking-quality metrics (`ndcg@k`, MRR, `recall@k`), zero extra
//!   dependencies.
//! - `serde` — `Serialize`/`Deserialize` on the core data model and
//!   explainability types.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod core;
#[cfg(feature = "eval")]
pub mod eval;
pub mod explain;
pub mod fusion;
pub mod normalization;
pub mod reranking;

pub use crate::core::{Candidate, FirstWins, MergePolicy, RankedList, Scored, ScoredList};
pub use crate::explain::{Explained, SourceContribution};
pub use crate::fusion::{Fusion, FusionError, LinearFusion, Rrf, WeightedRrf};
pub use crate::normalization::{MinMax, Normalizer, Softmax, ZScore};
pub use crate::reranking::{Pipeline, Reranker, TopK};
