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
//! External sources ──▶ Vec<RankedList> ──▶ fusion ──▶ reranking ──▶ final ranking
//! ```
//!
//! ## Status
//!
//! `0.0.x` — API under active design, expect breaking changes until `0.1.0`.
//! The roadmap lives in the [repository issues](https://github.com/Lsh0x/rankfusion/issues).

/// A single candidate in a ranked list.
///
/// `Id` is generic (`u64`, UUIDs, strings, custom types) — the library never
/// interprets it beyond equality and hashing. `Metadata` is opaque payload
/// carried through the pipeline untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate<Id, Metadata = ()> {
    pub id: Id,
    pub score: f32,
    pub metadata: Metadata,
}

impl<Id, Metadata> Candidate<Id, Metadata> {
    pub fn new(id: Id, score: f32, metadata: Metadata) -> Self {
        Self {
            id,
            score,
            metadata,
        }
    }
}

/// An ordered list of candidates produced by one external system.
///
/// Rank is implicit: position 0 is rank 1. Lists may have different lengths,
/// contain candidates missing from other lists, or be empty.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RankedList<Id, Metadata = ()> {
    pub candidates: Vec<Candidate<Id, Metadata>>,
}

impl<Id, Metadata> RankedList<Id, Metadata> {
    pub fn new(candidates: Vec<Candidate<Id, Metadata>>) -> Self {
        Self { candidates }
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_generic_ids() {
        let a = Candidate::new(42u64, 0.9, ());
        let b = Candidate::new("doc-1".to_string(), 0.5, ());
        assert_eq!(a.id, 42);
        assert_eq!(b.id, "doc-1");
    }

    #[test]
    fn ranked_list_basics() {
        let list: RankedList<u64> = RankedList::default();
        assert!(list.is_empty());
        let list = RankedList::new(vec![Candidate::new(1u64, 1.0, ())]);
        assert_eq!(list.len(), 1);
    }
}
