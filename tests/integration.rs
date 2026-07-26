//! End-to-end tests against the public API, as an external consumer sees it.

use rankfusion::{
    Candidate, LinearFusion, MinMax, Normalizer, Pipeline, RankedList, Rrf, Scored, ScoredList,
    Softmax, TopK, WeightedRrf, ZScore,
};

fn vector_results() -> RankedList<&'static str> {
    ["doc-1", "doc-2", "doc-3"].into_iter().collect()
}

fn keyword_results() -> RankedList<&'static str> {
    ["doc-2", "doc-4", "doc-1"].into_iter().collect()
}

#[test]
fn hybrid_search_scenario_end_to_end() {
    let results = Pipeline::new(Rrf::default())
        .reranker(TopK::new(2))
        .rank(vec![vector_results(), keyword_results()])
        .unwrap();

    assert_eq!(results.len(), 2);
    // doc-2 appears at rank 2 and rank 1 — the best combined evidence
    assert_eq!(*results[0].id(), "doc-2");
    assert!(results[0].score >= results[1].score);
}

#[test]
fn fusion_output_is_the_union_of_inputs() {
    let fused = Rrf::default().fuse(vec![vector_results(), keyword_results()]);
    let mut ids: Vec<&str> = fused.iter().map(|s| *s.id()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["doc-1", "doc-2", "doc-3", "doc-4"]);
}

#[test]
fn weight_count_mismatch_is_reported_not_silently_truncated() {
    let err = WeightedRrf::new(60.0, vec![1.0])
        .fuse(vec![vector_results(), keyword_results()])
        .unwrap_err();
    assert!(err.to_string().contains("weight count mismatch"));
}

#[test]
fn score_fusion_over_heterogeneous_scales() {
    // one source scores in [0, 1], the other in the hundreds — normalization
    // is what makes the linear combination meaningful
    let semantic = ScoredList::new(vec![
        Scored::new("doc-1", 0.91, ()),
        Scored::new("doc-2", 0.85, ()),
        Scored::new("doc-4", 0.10, ()),
    ]);
    let bm25 = ScoredList::new(vec![
        Scored::new("doc-2", 812.0, ()),
        Scored::new("doc-3", 210.0, ()),
    ]);

    let fused = LinearFusion::new(vec![1.0, 1.0])
        .fuse(vec![semantic, bm25])
        .unwrap();

    // doc-2 is strong in both sources; doc-1 is strong in one only. The raw
    // magnitudes (0.85 vs 812.0) never enter the comparison — normalization
    // per source is what makes the sum meaningful.
    assert_eq!(fused.len(), 4);
    assert_eq!(*fused[0].id(), "doc-2");
    assert!(fused[0].score > fused[1].score);
}

#[test]
fn metadata_survives_fusion_under_first_wins() {
    let with_meta: RankedList<&str, &str> = RankedList::new(vec![
        Candidate::new("doc-1", "from-vector"),
        Candidate::new("doc-2", "from-vector"),
    ]);
    let other: RankedList<&str, &str> = RankedList::new(vec![Candidate::new("doc-1", "from-bm25")]);

    let fused = Rrf::default().fuse(vec![with_meta, other]);
    let doc1 = fused.iter().find(|s| *s.id() == "doc-1").unwrap();
    assert_eq!(doc1.candidate.metadata, "from-vector");
}

#[test]
fn custom_merge_policy_is_applied() {
    let a: RankedList<&str, u32> = RankedList::new(vec![Candidate::new("doc-1", 3)]);
    let b: RankedList<&str, u32> = RankedList::new(vec![Candidate::new("doc-1", 7)]);

    let max_wins = |kept: &mut u32, incoming: u32| *kept = (*kept).max(incoming);
    let fused = Rrf::default().fuse_merge(vec![a, b], &max_wins);
    assert_eq!(fused[0].candidate.metadata, 7);
}

#[test]
fn explained_contributions_sum_to_the_fused_score() {
    let explained = Rrf::default().fuse_explained(vec![vector_results(), keyword_results()]);

    for result in &explained {
        let sum: f32 = result.contributions.iter().map(|c| c.partial_score).sum();
        assert!(
            (sum - result.score()).abs() < 1e-6,
            "{}: contributions {sum} != fused {}",
            result.id(),
            result.score()
        );
    }
}

#[test]
fn empty_input_yields_empty_output() {
    let fused = Rrf::default().fuse(Vec::<RankedList<&str>>::new());
    assert!(fused.is_empty());

    let fused = Rrf::default().fuse(vec![RankedList::<&str>::default()]);
    assert!(fused.is_empty());
}

#[test]
fn normalizers_are_usable_standalone() {
    let mut scores = [1.0f32, 3.0, 5.0];
    MinMax.normalize(&mut scores);
    assert_eq!(scores, [0.0, 0.5, 1.0]);

    let mut scores = [1.0f32, 2.0, 3.0];
    ZScore.normalize(&mut scores);
    assert!((scores[1]).abs() < 1e-6, "mean maps to 0");

    let mut scores = [1.0f32, 2.0, 3.0];
    Softmax.normalize(&mut scores);
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax sums to 1, got {sum}");
}

#[cfg(feature = "eval")]
#[test]
fn eval_metrics_compare_two_fusion_configs() {
    use rankfusion::eval::{ndcg_at_k, recall_at_k, reciprocal_rank};
    use std::collections::{HashMap, HashSet};

    let relevance: HashMap<&str, f32> = [("doc-2", 3.0), ("doc-1", 1.0)].into_iter().collect();
    let relevant: HashSet<&str> = ["doc-1", "doc-2"].into_iter().collect();

    let tight = Rrf::new(20.0).fuse(vec![vector_results(), keyword_results()]);
    let ids: Vec<&str> = tight.iter().map(|s| *s.id()).collect();

    let ndcg = ndcg_at_k(&ids, &relevance, 10);
    assert!((0.0..=1.0).contains(&ndcg), "ndcg out of range: {ndcg}");
    assert_eq!(reciprocal_rank(&ids, &relevant), 1.0, "doc-2 ranks first");
    assert_eq!(recall_at_k(&ids, &relevant, 10), 1.0);
}

#[cfg(feature = "serde")]
#[test]
fn core_types_round_trip_through_json() {
    let scored = Scored::new("doc-1".to_string(), 0.75, "payload".to_string());
    let json = serde_json::to_string(&scored).unwrap();
    let back: Scored<String, String> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, scored);

    let list: RankedList<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
    let json = serde_json::to_string(&list).unwrap();
    let back: RankedList<String> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, list);

    let scored_list = ScoredList::new(vec![Scored::new(1u64, 0.5, ())]);
    let json = serde_json::to_string(&scored_list).unwrap();
    let back: ScoredList<u64> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, scored_list);
}

#[cfg(feature = "serde")]
#[test]
fn explained_results_round_trip_through_json() {
    use rankfusion::Explained;

    let explained = Rrf::default().fuse_explained(vec![vector_results(), keyword_results()]);
    let owned: Vec<Explained<String>> = explained
        .iter()
        .map(|e| Explained {
            scored: Scored::new(e.id().to_string(), e.score(), ()),
            contributions: e.contributions.clone(),
        })
        .collect();

    let json = serde_json::to_string(&owned).unwrap();
    let back: Vec<Explained<String>> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, owned);
}
