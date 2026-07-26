//! Property-based tests: invariants that must hold for *any* input.
//!
//! Hand-written tests pin down the cases we thought of; these pin down the
//! ones we did not — degenerate lists, duplicate ids across sources, NaN and
//! infinite scores, and the float accumulation order of the fusion loop.

use std::collections::{HashMap, HashSet};

use proptest::prelude::*;
use rankfusion::{MinMax, Normalizer, Pipeline, RankedList, Rrf, Scored, Softmax, TopK, ZScore};

/// Ids for one source list: deduplicated, order preserved (a source does not
/// return the same document twice).
fn source_list() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..40, 0..25).prop_map(|ids| {
        let mut seen = HashSet::new();
        ids.into_iter().filter(|id| seen.insert(*id)).collect()
    })
}

fn source_lists() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(source_list(), 0..5)
}

/// Finite scores in a range where `f32` accumulation stays well-conditioned.
fn finite_scores() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1_000.0f32..1_000.0, 0..30)
}

fn to_ranked(lists: &[Vec<u8>]) -> Vec<RankedList<u8>> {
    lists
        .iter()
        .map(|ids| ids.iter().copied().collect())
        .collect()
}

fn score_map(fused: &[Scored<u8>]) -> HashMap<u8, f32> {
    fused.iter().map(|s| (*s.id(), s.score)).collect()
}

proptest! {
    /// Fusion neither invents nor loses candidates: the output ids are exactly
    /// the union of the input ids, each appearing once.
    #[test]
    fn rrf_output_is_the_deduplicated_union_of_inputs(lists in source_lists()) {
        let expected: HashSet<u8> = lists.iter().flatten().copied().collect();
        let fused = Rrf::default().fuse(to_ranked(&lists));

        let got: HashSet<u8> = fused.iter().map(|s| *s.id()).collect();
        prop_assert_eq!(got.len(), fused.len(), "an id appears twice in the output");
        prop_assert_eq!(got, expected);
    }

    /// The output is always sorted by descending score under the crate's total
    /// ordering — no input can produce an unsorted ranking.
    #[test]
    fn rrf_output_is_sorted_by_descending_score(lists in source_lists()) {
        let fused = Rrf::default().fuse(to_ranked(&lists));
        for pair in fused.windows(2) {
            prop_assert!(
                pair[0].score.total_cmp(&pair[1].score).is_ge(),
                "{} then {}", pair[0].score, pair[1].score
            );
        }
    }

    /// Reordering the input lists is not observable in the scores: RRF sums
    /// per-list contributions, and addition is commutative (up to `f32`
    /// accumulation error).
    #[test]
    fn rrf_scores_are_invariant_under_input_list_permutation(
        lists in source_lists(),
        rotation in 0usize..5,
    ) {
        prop_assume!(!lists.is_empty());
        let mut permuted = lists.clone();
        permuted.rotate_left(rotation % lists.len());

        let a = score_map(&Rrf::default().fuse(to_ranked(&lists)));
        let b = score_map(&Rrf::default().fuse(to_ranked(&permuted)));

        prop_assert_eq!(a.len(), b.len());
        for (id, score) in &a {
            let other = b[id];
            prop_assert!(
                (score - other).abs() <= 1e-5 * score.abs().max(1.0),
                "id {}: {} vs {}", id, score, other
            );
        }
    }

    /// With a single input list, fusion is order-preserving: RRF is strictly
    /// decreasing in rank, so the source's order survives untouched.
    #[test]
    fn rrf_over_a_single_list_preserves_its_order(ids in source_list()) {
        let fused = Rrf::default().fuse(to_ranked(std::slice::from_ref(&ids)));
        let got: Vec<u8> = fused.iter().map(|s| *s.id()).collect();
        prop_assert_eq!(got, ids);
    }

    /// Every RRF score is strictly positive and finite: `1 / (k + rank)` with
    /// `k = 60` and `rank >= 1` can neither vanish nor blow up.
    #[test]
    fn rrf_scores_are_finite_and_positive(lists in source_lists()) {
        let fused = Rrf::default().fuse(to_ranked(&lists));
        for scored in &fused {
            prop_assert!(scored.score.is_finite(), "non-finite score {}", scored.score);
            prop_assert!(scored.score > 0.0, "non-positive score {}", scored.score);
        }
    }

    /// The explainability path agrees with the plain path — same results, and
    /// the per-source contributions sum back to the fused score.
    #[test]
    fn explained_contributions_reconstruct_the_fused_score(lists in source_lists()) {
        let explained = Rrf::default().fuse_explained(to_ranked(&lists));
        let plain = Rrf::default().fuse(to_ranked(&lists));

        prop_assert_eq!(explained.len(), plain.len());
        for (e, p) in explained.iter().zip(&plain) {
            prop_assert_eq!(e.id(), p.id());
            prop_assert!(!e.contributions.is_empty(), "a result with no source");
            let sum: f32 = e.contributions.iter().map(|c| c.partial_score).sum();
            prop_assert!(
                (sum - e.score()).abs() <= 1e-5 * e.score().abs().max(1.0),
                "contributions {} != fused {}", sum, e.score()
            );
        }
    }

    /// Sorting never panics and stays total, even fed NaN and infinities —
    /// the crate-wide `total_cmp` policy.
    #[test]
    fn scores_including_nan_and_infinity_sort_deterministically(
        raw in prop::collection::vec(
            prop_oneof![
                Just(f32::NAN),
                Just(f32::INFINITY),
                Just(f32::NEG_INFINITY),
                -1e6f32..1e6,
            ],
            0..20,
        ),
    ) {
        let mut items: Vec<Scored<usize>> = raw
            .iter()
            .enumerate()
            .map(|(i, score)| Scored::new(i, *score, ()))
            .collect();

        items.sort_by(Scored::cmp_score_desc);
        let first: Vec<usize> = items.iter().map(|s| *s.id()).collect();
        items.sort_by(Scored::cmp_score_desc);
        let second: Vec<usize> = items.iter().map(|s| *s.id()).collect();
        prop_assert_eq!(first, second, "re-sorting changed the order");
    }

    /// Min-max always lands inside `[0, 1]`, including the degenerate cases
    /// (empty, single element, all-equal) documented as mapping to `1.0`.
    #[test]
    fn minmax_maps_into_the_unit_interval(scores in finite_scores()) {
        let mut normalized = scores.clone();
        MinMax.normalize(&mut normalized);

        prop_assert_eq!(normalized.len(), scores.len());
        for value in &normalized {
            prop_assert!((0.0..=1.0).contains(value), "out of range: {}", value);
        }
    }

    /// Normalizing an already-normalized slice changes nothing: min-max is
    /// idempotent because `[0, 1]` is its own image.
    #[test]
    fn minmax_is_idempotent(scores in finite_scores()) {
        let mut once = scores.clone();
        MinMax.normalize(&mut once);
        let mut twice = once.clone();
        MinMax.normalize(&mut twice);

        for (a, b) in once.iter().zip(&twice) {
            prop_assert!((a - b).abs() <= 1e-6, "{} != {}", a, b);
        }
    }

    /// Normalization is monotone: it may compress ties, never reorder. This is
    /// what makes it safe to apply per source before fusing.
    #[test]
    fn normalizers_preserve_ranking_order(scores in finite_scores()) {
        for (label, normalizer) in [
            ("minmax", &MinMax as &dyn Normalizer),
            ("zscore", &ZScore),
            ("softmax", &Softmax),
        ] {
            let mut normalized = scores.clone();
            normalizer.normalize(&mut normalized);

            for i in 0..scores.len() {
                for j in 0..scores.len() {
                    if scores[i] < scores[j] {
                        prop_assert!(
                            normalized[i] <= normalized[j],
                            "{}: {} < {} but {} > {}",
                            label, scores[i], scores[j], normalized[i], normalized[j]
                        );
                    }
                }
            }
        }
    }

    /// Softmax output is a probability distribution: non-negative and summing
    /// to one (empty slices excepted — there is nothing to distribute).
    #[test]
    fn softmax_yields_a_probability_distribution(scores in finite_scores()) {
        prop_assume!(!scores.is_empty());
        let mut normalized = scores.clone();
        Softmax.normalize(&mut normalized);

        for value in &normalized {
            prop_assert!(*value >= 0.0, "negative probability {}", value);
        }
        let sum: f32 = normalized.iter().sum();
        prop_assert!((sum - 1.0).abs() <= 1e-4, "sum is {}", sum);
    }

    /// `TopK` truncates without reordering: the result is the k-prefix of what
    /// the previous stage produced.
    #[test]
    fn topk_returns_the_prefix_of_the_fused_ranking(lists in source_lists(), k in 0usize..30) {
        let full = Pipeline::new(Rrf::default()).rank(to_ranked(&lists)).unwrap();
        let truncated = Pipeline::new(Rrf::default())
            .reranker(TopK::new(k))
            .rank(to_ranked(&lists))
            .unwrap();

        prop_assert_eq!(truncated.len(), k.min(full.len()));
        for (a, b) in truncated.iter().zip(&full) {
            prop_assert_eq!(a.id(), b.id());
        }
    }
}
