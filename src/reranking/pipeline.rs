//! The pipeline: one fusion strategy followed by ordered reranking stages.

use super::Reranker;
use crate::core::Scored;
use crate::fusion::{Fusion, FusionError};

/// A fusion strategy followed by reranking stages, run in declaration order.
///
/// The input list type is dictated by the fusion strategy's
/// [`Fusion::Input`] associated type — a `Pipeline<Rrf>` ranks
/// `Vec<RankedList>`, a `Pipeline<LinearFusion>` ranks `Vec<ScoredList>`,
/// and the compiler rejects any mix-up. Stages are boxed
/// (`Box<dyn Reranker>`): one allocation per stage at build time, negligible
/// next to the fusion sort.
///
/// ```
/// use rankfusion::{Pipeline, RankedList, Rrf, TopK};
///
/// let list_a: RankedList<&str> = ["a", "b", "c"].into_iter().collect();
/// let list_b: RankedList<&str> = ["b", "d", "a"].into_iter().collect();
///
/// let results = Pipeline::new(Rrf::default())
///     .reranker(TopK::new(2))
///     .rank(vec![list_a, list_b])
///     .unwrap();
///
/// assert_eq!(results.len(), 2);
/// assert_eq!(*results[0].id(), "b");
/// ```
pub struct Pipeline<F, Id, Metadata> {
    fusion: F,
    stages: Vec<Box<dyn Reranker<Id, Metadata>>>,
}

impl<F, Id, Metadata> Pipeline<F, Id, Metadata>
where
    F: Fusion<Id, Metadata>,
{
    pub fn new(fusion: F) -> Self {
        Self {
            fusion,
            stages: Vec::new(),
        }
    }

    /// Append a reranking stage. Stages run in the order they were added.
    #[must_use]
    pub fn reranker(mut self, stage: impl Reranker<Id, Metadata> + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Fuse the input lists, then run every stage in order.
    pub fn rank(&self, lists: Vec<F::Input>) -> Result<Vec<Scored<Id, Metadata>>, FusionError> {
        let mut candidates = self.fusion.fuse(lists)?;
        for stage in &self.stages {
            stage.rerank(&mut candidates);
        }
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{RankedList, ScoredList};
    use crate::fusion::{LinearFusion, Rrf, WeightedRrf};

    fn lists() -> Vec<RankedList<&'static str>> {
        vec![
            ["a", "b", "c"].into_iter().collect(),
            ["b", "d", "a"].into_iter().collect(),
        ]
    }

    fn ids<Id: Copy, M>(fused: &[Scored<Id, M>]) -> Vec<Id> {
        fused.iter().map(|s| *s.id()).collect()
    }

    #[test]
    fn stages_run_in_declaration_order() {
        let boost_c = |candidates: &mut Vec<Scored<&str, ()>>| {
            for c in candidates.iter_mut() {
                if *c.id() == "c" {
                    c.score += 10.0;
                }
            }
            candidates.sort_by(Scored::cmp_score_desc);
        };

        // boost then truncate: c survives
        let boosted_first = Pipeline::new(Rrf::default())
            .reranker(boost_c)
            .reranker(super::super::TopK::new(2))
            .rank(lists())
            .unwrap();
        assert_eq!(ids(&boosted_first), ["c", "b"]);

        // truncate then boost: c is already gone — order matters
        let truncated_first = Pipeline::new(Rrf::default())
            .reranker(super::super::TopK::new(2))
            .reranker(boost_c)
            .rank(lists())
            .unwrap();
        assert_eq!(ids(&truncated_first), ["b", "a"]);
    }

    #[test]
    fn topk_truncates_via_mut_vec() {
        let results = Pipeline::new(Rrf::default())
            .reranker(super::super::TopK::new(1))
            .rank(lists())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].id(), "b");
    }

    #[test]
    fn pipeline_without_stages_is_plain_fusion() {
        let piped = Pipeline::new(Rrf::default()).rank(lists()).unwrap();
        let direct = Rrf::default().fuse(lists());
        assert_eq!(piped, direct);
    }

    #[test]
    fn linear_fusion_pipelines_consume_scored_lists() {
        let vector: ScoredList<&str> = ScoredList::new(vec![
            Scored::new("a", 0.9, ()),
            Scored::new("b", 0.7, ()),
            Scored::new("c", 0.1, ()),
        ]);
        let results = Pipeline::new(LinearFusion::new(vec![1.0]))
            .reranker(super::super::TopK::new(2))
            .rank(vec![vector])
            .unwrap();
        assert_eq!(ids(&results), ["a", "b"]);
    }

    #[test]
    fn fusion_errors_propagate_through_rank() {
        let err = Pipeline::new(WeightedRrf::new(60.0, vec![1.0]))
            .rank(lists())
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
    fn filtering_stage_can_drop_candidates() {
        let drop_a = |candidates: &mut Vec<Scored<&str, ()>>| {
            candidates.retain(|c| *c.id() != "a");
        };
        let results = Pipeline::new(Rrf::default())
            .reranker(drop_a)
            .rank(lists())
            .unwrap();
        assert!(!ids(&results).contains(&"a"));
    }
}
