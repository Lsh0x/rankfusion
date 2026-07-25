//! Reciprocal Rank Fusion — plain and weighted.

use std::hash::Hash;

use super::{accumulate_rrf, FusionError};
use crate::core::{FirstWins, MergePolicy, RankedList, Scored};

/// Reciprocal Rank Fusion.
///
/// `score(c) = Σ_i 1 / (k + rank_i(c))` over every input list containing `c`,
/// with rank starting at 1. Candidates appearing high in several lists rank
/// first. `k` (default [`Rrf::DEFAULT_K`]) dampens the influence of top
/// ranks: larger `k` flattens the contribution curve.
///
/// ```
/// use rankfusion::{RankedList, Rrf};
///
/// let list_a: RankedList<&str> = ["a", "b", "c"].into_iter().collect();
/// let list_b: RankedList<&str> = ["b", "d", "a"].into_iter().collect();
///
/// let fused = Rrf::default().fuse(vec![list_a, list_b]);
/// // "b" appears strongly in both lists and wins
/// assert_eq!(*fused[0].id(), "b");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rrf {
    k: f32,
}

impl Rrf {
    /// The conventional default for `k`, from the original RRF paper.
    pub const DEFAULT_K: f32 = 60.0;

    pub fn new(k: f32) -> Self {
        Self { k }
    }

    pub fn k(&self) -> f32 {
        self.k
    }

    /// Fuse the lists into a single scored ranking, best first.
    ///
    /// Duplicate candidates across lists are merged with the [`FirstWins`]
    /// metadata policy; use [`Rrf::fuse_merge`] for a custom policy.
    pub fn fuse<Id, Metadata>(
        &self,
        lists: Vec<RankedList<Id, Metadata>>,
    ) -> Vec<Scored<Id, Metadata>>
    where
        Id: Eq + Hash + Clone,
    {
        self.fuse_merge(lists, &FirstWins)
    }

    /// [`Rrf::fuse`] with an explicit metadata [`MergePolicy`].
    pub fn fuse_merge<Id, Metadata, P>(
        &self,
        lists: Vec<RankedList<Id, Metadata>>,
        policy: &P,
    ) -> Vec<Scored<Id, Metadata>>
    where
        Id: Eq + Hash + Clone,
        P: MergePolicy<Metadata>,
    {
        accumulate_rrf(self.k, lists, None, policy)
    }
}

impl Default for Rrf {
    fn default() -> Self {
        Self::new(Self::DEFAULT_K)
    }
}

/// Weighted Reciprocal Rank Fusion.
///
/// `score(c) = Σ_i weight_i / (k + rank_i(c))`. The library does not
/// interpret the weights — they are user-defined per input list, in the same
/// order as the lists passed to [`WeightedRrf::fuse`]. A weight/list count
/// mismatch is a [`FusionError::WeightCountMismatch`], never a silent
/// truncation.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedRrf {
    k: f32,
    weights: Vec<f32>,
}

impl WeightedRrf {
    pub fn new(k: f32, weights: Vec<f32>) -> Self {
        Self { k, weights }
    }

    /// Fuse with per-list weights, best first. [`FirstWins`] metadata policy;
    /// use [`WeightedRrf::fuse_merge`] for a custom one.
    pub fn fuse<Id, Metadata>(
        &self,
        lists: Vec<RankedList<Id, Metadata>>,
    ) -> Result<Vec<Scored<Id, Metadata>>, FusionError>
    where
        Id: Eq + Hash + Clone,
    {
        self.fuse_merge(lists, &FirstWins)
    }

    /// [`WeightedRrf::fuse`] with an explicit metadata [`MergePolicy`].
    pub fn fuse_merge<Id, Metadata, P>(
        &self,
        lists: Vec<RankedList<Id, Metadata>>,
        policy: &P,
    ) -> Result<Vec<Scored<Id, Metadata>>, FusionError>
    where
        Id: Eq + Hash + Clone,
        P: MergePolicy<Metadata>,
    {
        if self.weights.len() != lists.len() {
            return Err(FusionError::WeightCountMismatch {
                expected: self.weights.len(),
                got: lists.len(),
            });
        }
        Ok(accumulate_rrf(self.k, lists, Some(&self.weights), policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Candidate;
    use std::collections::HashSet;

    fn ids<Id: Copy, M>(fused: &[Scored<Id, M>]) -> Vec<Id> {
        fused.iter().map(|s| *s.id()).collect()
    }

    #[test]
    fn textbook_example_b_ranks_first() {
        let list_a: RankedList<&str> = ["a", "b", "c"].into_iter().collect();
        let list_b: RankedList<&str> = ["b", "d", "a"].into_iter().collect();
        let fused = Rrf::default().fuse(vec![list_a, list_b]);
        // k=60: b = 1/62 + 1/61, a = 1/61 + 1/63, d = 1/62, c = 1/63
        assert_eq!(ids(&fused), ["b", "a", "d", "c"]);
    }

    #[test]
    fn rank_starts_at_one() {
        let list: RankedList<&str> = ["only"].into_iter().collect();
        let fused = Rrf::new(0.0).fuse(vec![list]);
        // rank 1, k 0 → score = 1/(0+1) = 1.0 exactly
        assert_eq!(fused[0].score, 1.0);
    }

    #[test]
    fn every_distinct_id_appears_exactly_once() {
        let list_a: RankedList<u64> = [1, 2, 3, 4].into_iter().collect();
        let list_b: RankedList<u64> = [3, 4, 5, 6].into_iter().collect();
        let list_c: RankedList<u64> = [6, 1].into_iter().collect();
        let fused = Rrf::default().fuse(vec![list_a, list_b, list_c]);
        let unique: HashSet<u64> = fused.iter().map(|s| *s.id()).collect();
        assert_eq!(fused.len(), unique.len());
        assert_eq!(unique, (1..=6).collect::<HashSet<u64>>());
    }

    #[test]
    fn empty_inputs_yield_empty_output() {
        let none: Vec<RankedList<u64>> = vec![];
        assert!(Rrf::default().fuse(none).is_empty());

        let empties: Vec<RankedList<u64>> = vec![RankedList::default(), RankedList::default()];
        assert!(Rrf::default().fuse(empties).is_empty());
    }

    #[test]
    fn empty_list_among_full_ones_is_harmless() {
        let list_a: RankedList<&str> = ["a"].into_iter().collect();
        let fused = Rrf::default().fuse(vec![RankedList::default(), list_a]);
        assert_eq!(ids(&fused), ["a"]);
    }

    #[test]
    fn unit_weights_match_plain_rrf() {
        let mk = || -> Vec<RankedList<&str>> {
            vec![
                ["a", "b", "c"].into_iter().collect(),
                ["b", "d", "a"].into_iter().collect(),
            ]
        };
        let plain = Rrf::default().fuse(mk());
        let weighted = WeightedRrf::new(Rrf::DEFAULT_K, vec![1.0, 1.0])
            .fuse(mk())
            .unwrap();
        assert_eq!(plain, weighted);
    }

    #[test]
    fn weight_count_mismatch_is_a_typed_error() {
        let lists: Vec<RankedList<u64>> = vec![[1u64].into_iter().collect()];
        let err = WeightedRrf::new(60.0, vec![1.0, 0.5])
            .fuse(lists)
            .unwrap_err();
        assert_eq!(
            err,
            FusionError::WeightCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn weights_shift_the_ranking() {
        let mk = || -> Vec<RankedList<&str>> {
            vec![
                ["a", "b"].into_iter().collect(),
                ["b", "a"].into_iter().collect(),
            ]
        };
        // symmetric lists: unit weights tie a and b; boosting list 1 puts b first
        let boosted = WeightedRrf::new(60.0, vec![1.0, 2.0]).fuse(mk()).unwrap();
        assert_eq!(ids(&boosted)[0], "b");
    }

    #[test]
    fn ties_break_by_first_seen_order_deterministically() {
        // x and y each appear once at rank 1 in different lists → equal score
        let lists: Vec<RankedList<&str>> =
            vec![["x"].into_iter().collect(), ["y"].into_iter().collect()];
        for _ in 0..8 {
            let fused = Rrf::default().fuse(lists.clone());
            assert_eq!(ids(&fused), ["x", "y"]);
        }
    }

    #[test]
    fn metadata_first_wins_by_default() {
        let list_a = RankedList::new(vec![Candidate::new("doc", "from-a")]);
        let list_b = RankedList::new(vec![Candidate::new("doc", "from-b")]);
        let fused = Rrf::default().fuse(vec![list_a, list_b]);
        assert_eq!(fused[0].candidate.metadata, "from-a");
    }

    #[test]
    fn custom_merge_policy_applies() {
        let list_a = RankedList::new(vec![Candidate::new("doc", 1u32)]);
        let list_b = RankedList::new(vec![Candidate::new("doc", 7u32)]);
        let max_wins = |kept: &mut u32, incoming: u32| *kept = (*kept).max(incoming);
        let fused = Rrf::default().fuse_merge(vec![list_a, list_b], &max_wins);
        assert_eq!(fused[0].candidate.metadata, 7);
    }
}
