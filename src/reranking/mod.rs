//! Post-fusion reranking: composable, synchronous, in-place stages.
//!
//! A [`Reranker`] mutates the fused candidate vector — boosting, filtering,
//! truncating, re-ordering. Stages run in declaration order inside a
//! [`Pipeline`]. The trait is deliberately synchronous: rerankers are CPU
//! operations; async systems (remote ML rerankers, APIs) adapt *outside* the
//! core by materializing their result before or after the pipeline.
//!
//! Stages own the output order: a stage that changes scores is responsible
//! for re-sorting if it wants score order restored (see the closure example
//! on [`Reranker`]).

mod pipeline;

pub use pipeline::Pipeline;

use crate::core::Scored;

/// A post-fusion stage that mutates the ranked candidates in place.
///
/// The receiver is `&mut Vec` (not `&mut [_]`) so stages may filter and
/// truncate, not just reorder. Any `Fn(&mut Vec<Scored<Id, M>>)` closure is a
/// reranker — no struct needed for one-liners:
///
/// ```
/// use rankfusion::{Pipeline, RankedList, Rrf, Scored};
///
/// let boost_doc_b = |candidates: &mut Vec<Scored<&str, ()>>| {
///     for c in candidates.iter_mut() {
///         if *c.id() == "b" {
///             c.score *= 2.0;
///         }
///     }
///     candidates.sort_by(Scored::cmp_score_desc);
/// };
///
/// let list: RankedList<&str> = ["a", "b"].into_iter().collect();
/// let fused = Pipeline::new(Rrf::default())
///     .reranker(boost_doc_b)
///     .rank(vec![list])
///     .unwrap();
/// assert_eq!(*fused[0].id(), "b");
/// ```
pub trait Reranker<Id, Metadata> {
    /// Mutate the fused candidates in place: reorder, rescore, or truncate.
    fn rerank(&self, candidates: &mut Vec<Scored<Id, Metadata>>);
}

impl<Id, Metadata, F> Reranker<Id, Metadata> for F
where
    F: Fn(&mut Vec<Scored<Id, Metadata>>),
{
    fn rerank(&self, candidates: &mut Vec<Scored<Id, Metadata>>) {
        self(candidates);
    }
}

/// Keep only the top `k` candidates (in current order).
///
/// The canonical truncating stage — final result sizing belongs at the end of
/// a pipeline, after boosts have settled the order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopK {
    k: usize,
}

impl TopK {
    /// A stage keeping only the first `k` candidates.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl<Id, Metadata> Reranker<Id, Metadata> for TopK {
    fn rerank(&self, candidates: &mut Vec<Scored<Id, Metadata>>) {
        candidates.truncate(self.k);
    }
}
