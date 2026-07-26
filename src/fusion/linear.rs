//! Linear score fusion over normalized per-list scores.

use std::hash::Hash;

use super::{accumulate_contributions, FusionError};
use crate::core::{FirstWins, MergePolicy, Scored, ScoredList};
use crate::normalization::{MinMax, Normalizer};

/// Linear score fusion: `score(c) = Σ_i weight_i · normalize(score_i(c))`.
///
/// Each input list's scores are normalized independently (**per source list,
/// never across the fused pool** — see the [`crate::normalization`] module
/// docs), then combined with per-list weights. [`MinMax`] is the default
/// normalizer; any [`Normalizer`] plugs in through
/// [`LinearFusion::with_normalizer`].
///
/// ```
/// use rankfusion::{LinearFusion, Scored, ScoredList};
///
/// // cosine similarity in [-1, 1] and a BM25-ish unbounded score:
/// let vector = ScoredList::new(vec![
///     Scored::new("a", 0.95, ()),
///     Scored::new("b", 0.85, ()),
///     Scored::new("d", 0.20, ()),
/// ]);
/// let keyword = ScoredList::new(vec![
///     Scored::new("b", 42.0, ()),
///     Scored::new("c", 17.0, ()),
/// ]);
///
/// let fused = LinearFusion::new(vec![1.0, 1.0])
///     .fuse(vec![vector, keyword])
///     .unwrap();
/// // "b" scores high in both normalized lists and wins
/// assert_eq!(*fused[0].id(), "b");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LinearFusion<N = MinMax> {
    weights: Vec<f32>,
    normalizer: N,
}

impl LinearFusion<MinMax> {
    /// Linear fusion with [`MinMax`] normalization.
    pub fn new(weights: Vec<f32>) -> Self {
        Self::with_normalizer(weights, MinMax)
    }
}

impl<N: Normalizer> LinearFusion<N> {
    /// Linear fusion with a custom [`Normalizer`].
    pub fn with_normalizer(weights: Vec<f32>, normalizer: N) -> Self {
        Self {
            weights,
            normalizer,
        }
    }

    /// Fuse the scored lists into a single ranking, best first.
    ///
    /// Duplicate candidates across lists are merged with the [`FirstWins`]
    /// metadata policy; use [`LinearFusion::fuse_merge`] for a custom policy.
    /// A weight/list count mismatch is a typed error, never a silent
    /// truncation.
    pub fn fuse<Id, Metadata>(
        &self,
        lists: Vec<ScoredList<Id, Metadata>>,
    ) -> Result<Vec<Scored<Id, Metadata>>, FusionError>
    where
        Id: Eq + Hash + Clone,
    {
        self.fuse_merge(lists, &FirstWins)
    }

    /// [`LinearFusion::fuse`] with an explicit metadata [`MergePolicy`].
    pub fn fuse_merge<Id, Metadata, P>(
        &self,
        lists: Vec<ScoredList<Id, Metadata>>,
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

        // one scratch buffer reused across lists — normalization needs a
        // contiguous &mut [f32] slice, scores live inside Scored structs
        let mut scratch: Vec<f32> = Vec::new();
        let mut contributions = Vec::new();
        for (list, &weight) in lists.into_iter().zip(&self.weights) {
            scratch.clear();
            scratch.extend(list.items.iter().map(|s| s.score));
            self.normalizer.normalize(&mut scratch);
            contributions.extend(
                list.items
                    .into_iter()
                    .zip(&scratch)
                    .map(|(scored, &norm)| (scored.candidate, weight * norm)),
            );
        }
        Ok(accumulate_contributions(contributions, policy))
    }
}

impl<N: Normalizer> LinearFusion<N> {
    /// [`LinearFusion::fuse`] with per-source contribution tracing — see
    /// [`crate::explain`]. Separate accumulation path: `fuse` stays untouched.
    pub fn fuse_explained<Id, Metadata>(
        &self,
        lists: Vec<ScoredList<Id, Metadata>>,
    ) -> Result<Vec<crate::explain::Explained<Id, Metadata>>, FusionError>
    where
        Id: Eq + Hash + Clone,
    {
        if self.weights.len() != lists.len() {
            return Err(FusionError::WeightCountMismatch {
                expected: self.weights.len(),
                got: lists.len(),
            });
        }

        let mut scratch: Vec<f32> = Vec::new();
        let mut entries = Vec::new();
        for (list_index, (list, &weight)) in lists.into_iter().zip(&self.weights).enumerate() {
            scratch.clear();
            scratch.extend(list.items.iter().map(|s| s.score));
            self.normalizer.normalize(&mut scratch);
            entries.extend(list.items.into_iter().zip(&scratch).enumerate().map(
                |(position, (scored, &norm))| {
                    (
                        scored.candidate,
                        crate::explain::SourceContribution {
                            list_index,
                            rank: position + 1,
                            partial_score: weight * norm,
                        },
                    )
                },
            ));
        }
        Ok(super::accumulate_explained(entries, &FirstWins))
    }
}

impl<Id, Metadata, N> super::Fusion<Id, Metadata> for LinearFusion<N>
where
    Id: Eq + Hash + Clone,
    N: Normalizer,
{
    type Input = ScoredList<Id, Metadata>;

    fn fuse(&self, lists: Vec<Self::Input>) -> Result<Vec<Scored<Id, Metadata>>, FusionError> {
        LinearFusion::fuse(self, lists)
    }

    fn fuse_explained(
        &self,
        lists: Vec<Self::Input>,
    ) -> Result<Vec<crate::explain::Explained<Id, Metadata>>, FusionError> {
        LinearFusion::fuse_explained(self, lists)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalization::{Softmax, ZScore};

    fn ids<Id: Copy, M>(fused: &[Scored<Id, M>]) -> Vec<Id> {
        fused.iter().map(|s| *s.id()).collect()
    }

    fn list(pairs: &[(&'static str, f32)]) -> ScoredList<&'static str> {
        ScoredList::new(
            pairs
                .iter()
                .map(|&(id, s)| Scored::new(id, s, ()))
                .collect(),
        )
    }

    #[test]
    fn incompatible_ranges_combine_after_normalization() {
        let cosine = list(&[("a", 0.95), ("b", 0.80)]);
        let bm25 = list(&[("b", 42.0), ("c", 17.0)]);
        let fused = LinearFusion::new(vec![1.0, 1.0])
            .fuse(vec![cosine, bm25])
            .unwrap();
        // minmax per list: a=1.0, b=0.0 | b=1.0, c=0.0 → a and b tie at 1.0,
        // first-seen breaks the tie in favour of a
        assert_eq!(ids(&fused)[..2], ["a", "b"]);
    }

    #[test]
    fn normalization_is_per_list_not_pooled() {
        // both lists are identical up to a x100 scale factor; per-list
        // normalization makes their contributions identical
        let small = list(&[("a", 0.1), ("b", 0.2)]);
        let large = list(&[("a", 10.0), ("b", 20.0)]);
        let fused = LinearFusion::new(vec![1.0, 1.0])
            .fuse(vec![small, large])
            .unwrap();
        assert_eq!(fused[0].score, 2.0); // b: 1.0 + 1.0
        assert_eq!(fused[1].score, 0.0); // a: 0.0 + 0.0
    }

    #[test]
    fn weights_scale_contributions() {
        let l1 = list(&[("a", 1.0), ("b", 0.0)]);
        let l2 = list(&[("b", 1.0), ("a", 0.0)]);
        // heavier weight on l2 puts b first despite symmetry
        let fused = LinearFusion::new(vec![1.0, 3.0])
            .fuse(vec![l1, l2])
            .unwrap();
        assert_eq!(ids(&fused)[0], "b");
        assert_eq!(fused[0].score, 3.0);
    }

    #[test]
    fn weight_count_mismatch_is_a_typed_error() {
        let err = LinearFusion::new(vec![1.0])
            .fuse(vec![list(&[("a", 1.0)]), list(&[("b", 1.0)])])
            .unwrap_err();
        assert_eq!(
            err,
            FusionError::WeightCountMismatch {
                expected: 1,
                got: 2
            }
        );
    }

    #[test]
    fn custom_normalizers_plug_in() {
        let l = list(&[("a", 1.0), ("b", 3.0), ("c", 2.0)]);
        let fused = LinearFusion::with_normalizer(vec![1.0], Softmax)
            .fuse(vec![l.clone()])
            .unwrap();
        let total: f32 = fused.iter().map(|s| s.score).sum();
        assert!((total - 1.0).abs() < 1e-6);

        let fused = LinearFusion::with_normalizer(vec![1.0], ZScore)
            .fuse(vec![l])
            .unwrap();
        assert_eq!(ids(&fused), ["b", "c", "a"]);
    }

    #[test]
    fn empty_lists_are_harmless() {
        let fused = LinearFusion::new(vec![1.0, 1.0])
            .fuse(vec![ScoredList::<&str>::default(), list(&[("a", 5.0)])])
            .unwrap();
        assert_eq!(ids(&fused), ["a"]);
        assert_eq!(fused[0].score, 1.0); // single element → minmax 1.0
    }

    #[test]
    fn metadata_merges_first_wins() {
        let l1 = ScoredList::new(vec![Scored::new("doc", 1.0, "from-1")]);
        let l2 = ScoredList::new(vec![Scored::new("doc", 1.0, "from-2")]);
        let fused = LinearFusion::new(vec![1.0, 1.0])
            .fuse(vec![l1, l2])
            .unwrap();
        assert_eq!(fused[0].candidate.metadata, "from-1");
    }

    #[test]
    fn candidate_order_within_list_does_not_matter_only_scores() {
        // a source may rank against its own scores; linear fusion trusts the
        // scores, not the positions
        let l = ScoredList::new(vec![
            Scored::new("low-first", 0.1, ()),
            Scored::new("high-second", 0.9, ()),
        ]);
        let fused = LinearFusion::new(vec![1.0]).fuse(vec![l]).unwrap();
        assert_eq!(ids(&fused)[0], "high-second");
    }
}
