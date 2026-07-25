//! Explainability: per-source contribution tracing.
//!
//! The first question anyone asks of a fused ranking is *"why is this result
//! first?"*. [`Explained`] answers it: for each result, one
//! [`SourceContribution`] per input list that contained the candidate — which
//! list, at what rank, contributing how much. The partial scores always sum
//! to the fused score (within `f32` accumulation error), for every fusion
//! strategy.
//!
//! Explanation is a **fusion-level** feature: it runs through a separate
//! accumulation path (`rank()`/`fuse()` are untouched by construction, zero
//! hot-path overhead), and [`crate::Pipeline::rank_explained`] deliberately
//! does *not* apply reranking stages — a stage mutating scores would break
//! the sum invariant, and boosts are the caller's own code.
//!
//! ```
//! use rankfusion::{RankedList, Rrf};
//!
//! let vector: RankedList<&str> = ["a", "b", "c"].into_iter().collect();
//! let keyword: RankedList<&str> = ["b", "d", "a"].into_iter().collect();
//!
//! let explained = Rrf::default().fuse_explained(vec![vector, keyword]);
//!
//! // why is "b" first? — rank 2 in list 0, rank 1 in list 1:
//! let b = &explained[0];
//! assert_eq!(*b.id(), "b");
//! let ranks: Vec<(usize, usize)> = b
//!     .contributions
//!     .iter()
//!     .map(|c| (c.list_index, c.rank))
//!     .collect();
//! assert_eq!(ranks, [(0, 2), (1, 1)]);
//! ```

use crate::core::Scored;

/// One input list's contribution to a fused result.
///
/// A contribution exists only for lists that actually contain the candidate,
/// so `rank` (1-based position in that list) is always known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceContribution {
    /// Index of the input list, in the order the lists were passed to fusion.
    pub list_index: usize,
    /// 1-based rank of the candidate within that list.
    pub rank: usize,
    /// This list's share of the fused score.
    pub partial_score: f32,
}

/// A fused result together with the per-source breakdown of its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Explained<Id, Metadata = ()> {
    pub scored: Scored<Id, Metadata>,
    /// One entry per input list containing this candidate, in first-seen
    /// order (list order, then rank).
    pub contributions: Vec<SourceContribution>,
}

impl<Id, Metadata> Explained<Id, Metadata> {
    pub fn id(&self) -> &Id {
        self.scored.id()
    }

    pub fn score(&self) -> f32 {
        self.scored.score
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{RankedList, Scored, ScoredList};
    use crate::fusion::{Fusion, LinearFusion, Rrf, WeightedRrf};
    use crate::reranking::{Pipeline, TopK};

    fn ranked() -> Vec<RankedList<&'static str>> {
        vec![
            ["a", "b", "c"].into_iter().collect(),
            ["b", "d", "a"].into_iter().collect(),
        ]
    }

    fn scored_lists() -> Vec<ScoredList<&'static str>> {
        vec![
            ScoredList::new(vec![
                Scored::new("a", 0.95, ()),
                Scored::new("b", 0.85, ()),
                Scored::new("d", 0.20, ()),
            ]),
            ScoredList::new(vec![Scored::new("b", 42.0, ()), Scored::new("c", 17.0, ())]),
        ]
    }

    #[track_caller]
    fn assert_matches_plain<Id, M>(explained: &[crate::Explained<Id, M>], plain: &[Scored<Id, M>])
    where
        Id: PartialEq + std::fmt::Debug,
        M: PartialEq + std::fmt::Debug,
    {
        assert_eq!(explained.len(), plain.len());
        for (e, p) in explained.iter().zip(plain) {
            assert_eq!(e.scored, *p);
            let sum: f32 = e.contributions.iter().map(|c| c.partial_score).sum();
            assert!(
                (sum - e.scored.score).abs() < 1e-6,
                "sum {sum} != fused {}",
                e.scored.score
            );
        }
    }

    #[test]
    fn rrf_contributions_sum_to_fused_score() {
        let explained = Rrf::default().fuse_explained(ranked());
        let plain = Rrf::default().fuse(ranked());
        assert_matches_plain(&explained, &plain);
    }

    #[test]
    fn weighted_rrf_contributions_sum_to_fused_score() {
        let w = WeightedRrf::new(60.0, vec![1.0, 2.5]);
        let explained = w.fuse_explained(ranked()).unwrap();
        let plain = w.fuse(ranked()).unwrap();
        assert_matches_plain(&explained, &plain);
    }

    #[test]
    fn linear_contributions_sum_to_fused_score() {
        let lf = LinearFusion::new(vec![1.0, 0.5]);
        let explained = lf.fuse_explained(scored_lists()).unwrap();
        let plain = lf.fuse(scored_lists()).unwrap();
        assert_matches_plain(&explained, &plain);
    }

    #[test]
    fn contributions_carry_list_and_rank() {
        let explained = Rrf::default().fuse_explained(ranked());
        let b = &explained[0];
        assert_eq!(*b.id(), "b");
        assert_eq!(b.contributions.len(), 2);
        assert_eq!(
            (b.contributions[0].list_index, b.contributions[0].rank),
            (0, 2)
        );
        assert_eq!(
            (b.contributions[1].list_index, b.contributions[1].rank),
            (1, 1)
        );
        // single-list candidate: exactly one contribution
        let c = explained.iter().find(|e| *e.id() == "c").unwrap();
        assert_eq!(c.contributions.len(), 1);
        assert_eq!(
            (c.contributions[0].list_index, c.contributions[0].rank),
            (0, 3)
        );
    }

    #[test]
    fn weighted_partial_scores_reflect_weights() {
        let w = WeightedRrf::new(0.0, vec![2.0, 1.0]);
        let explained = w.fuse_explained(ranked()).unwrap();
        let b = explained.iter().find(|e| *e.id() == "b").unwrap();
        // list 0 (w=2.0) rank 2 → 2/2 = 1.0 ; list 1 (w=1.0) rank 1 → 1/1 = 1.0
        assert_eq!(b.contributions[0].partial_score, 1.0);
        assert_eq!(b.contributions[1].partial_score, 1.0);
    }

    #[test]
    fn trait_fuse_explained_is_object_reachable() {
        fn generic<F: Fusion<&'static str, ()>>(f: &F, lists: Vec<F::Input>) -> usize {
            f.fuse_explained(lists).unwrap().len()
        }
        assert_eq!(generic(&Rrf::default(), ranked()), 4);
        assert_eq!(
            generic(&LinearFusion::new(vec![1.0, 1.0]), scored_lists()),
            4
        );
    }

    #[test]
    fn pipeline_rank_explained_ignores_stages() {
        // stages apply to rank() only — rank_explained explains the fusion
        let pipeline = Pipeline::new(Rrf::default()).reranker(TopK::new(1));
        let explained = pipeline.rank_explained(ranked()).unwrap();
        assert_eq!(explained.len(), 4); // TopK(1) NOT applied
        let plain = pipeline.rank(ranked()).unwrap();
        assert_eq!(plain.len(), 1); // TopK(1) applied
    }
}
