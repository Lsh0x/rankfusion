//! Offline ranking evaluation — `ndcg@k`, MRR, `recall@k`.
//!
//! Behind the `eval` cargo feature (off by default, zero extra
//! dependencies). Ranking metrics are pure ranking-layer concerns: use them
//! to compare fusion configurations against ground-truth judgments — RRF
//! `k=60` vs `k=20`, RRF vs linear, weight tuning — on your own data.
//!
//! Conventions: rankings are slices of ids, best first (what fusion
//! returns). Relevance judgments are either graded (`HashMap<Id, f32>`,
//! for NDCG) or binary (`HashSet<Id>`, for MRR / recall). Metrics return
//! `0.0` on degenerate inputs (no relevant documents) rather than NaN.
//!
//! ```
//! use std::collections::HashSet;
//! use rankfusion::eval::recall_at_k;
//!
//! let ranking = ["a", "b", "c"];
//! let relevant: HashSet<&str> = ["a", "c"].into_iter().collect();
//! assert_eq!(recall_at_k(&ranking, &relevant, 2), 0.5);
//! assert_eq!(recall_at_k(&ranking, &relevant, 3), 1.0);
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Normalized Discounted Cumulative Gain at `k`, with graded relevance.
///
/// `DCG@k = Σ_{i=1..k} gain(id_i) / log2(i + 1)`, normalized by the ideal
/// DCG (judgments sorted by gain, descending). Ids absent from `relevance`
/// gain `0.0`. Returns `0.0` when the ideal DCG is zero (no positive
/// judgments).
pub fn ndcg_at_k<Id>(ranking: &[Id], relevance: &HashMap<Id, f32>, k: usize) -> f32
where
    Id: Eq + Hash,
{
    let dcg: f32 = ranking
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| relevance.get(id).copied().unwrap_or(0.0) / (i as f32 + 2.0).log2())
        .sum();

    let mut ideal: Vec<f32> = relevance.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.total_cmp(a));
    let idcg: f32 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, gain)| gain / (i as f32 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Reciprocal rank of the first relevant result: `1 / rank`, or `0.0` if no
/// relevant document appears in the ranking.
pub fn reciprocal_rank<Id>(ranking: &[Id], relevant: &HashSet<Id>) -> f32
where
    Id: Eq + Hash,
{
    ranking
        .iter()
        .position(|id| relevant.contains(id))
        .map_or(0.0, |i| 1.0 / (i as f32 + 1.0))
}

/// Mean Reciprocal Rank over a set of queries — the mean of
/// [`reciprocal_rank`] over `(ranking, relevant)` pairs. Returns `0.0` for
/// an empty query set.
pub fn mrr<'a, Id>(queries: impl IntoIterator<Item = (&'a [Id], &'a HashSet<Id>)>) -> f32
where
    Id: Eq + Hash + 'a,
{
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for (ranking, relevant) in queries {
        sum += reciprocal_rank(ranking, relevant);
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

/// Fraction of the relevant documents present in the top `k`. Returns `0.0`
/// when `relevant` is empty.
pub fn recall_at_k<Id>(ranking: &[Id], relevant: &HashSet<Id>, k: usize) -> f32
where
    Id: Eq + Hash,
{
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranking
        .iter()
        .take(k)
        .filter(|id| relevant.contains(*id))
        .count();
    hits as f32 / relevant.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graded() -> HashMap<&'static str, f32> {
        [("a", 3.0), ("b", 2.0), ("c", 1.0)].into_iter().collect()
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        assert!((ndcg_at_k(&["a", "b", "c"], &graded(), 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ndcg_reversed_ranking_hand_computed() {
        // DCG  = 1/log2(2) + 2/log2(3) + 3/log2(4) = 1.0 + 1.26186 + 1.5 = 3.76186
        // IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3.0 + 1.26186 + 0.5 = 4.76186
        // NDCG = 3.76186 / 4.76186 = 0.79
        let ndcg = ndcg_at_k(&["c", "b", "a"], &graded(), 3);
        assert!((ndcg - 0.79).abs() < 1e-3, "got {ndcg}");
    }

    #[test]
    fn ndcg_cuts_at_k_and_ignores_unjudged() {
        // only the top-1 counts; "x" is unjudged → gain 0
        assert_eq!(ndcg_at_k(&["x", "a"], &graded(), 1), 0.0);
        assert!((ndcg_at_k(&["a", "x"], &graded(), 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ndcg_no_judgments_is_zero() {
        let empty: HashMap<&str, f32> = HashMap::new();
        assert_eq!(ndcg_at_k(&["a"], &empty, 3), 0.0);
    }

    #[test]
    fn reciprocal_rank_hand_computed() {
        let relevant: HashSet<&str> = ["a"].into_iter().collect();
        assert_eq!(reciprocal_rank(&["x", "a", "b"], &relevant), 0.5);
        assert_eq!(reciprocal_rank(&["a"], &relevant), 1.0);
        assert_eq!(reciprocal_rank(&["x", "y"], &relevant), 0.0);
    }

    #[test]
    fn mrr_averages_queries() {
        let rel_a: HashSet<&str> = ["a"].into_iter().collect();
        let q1: &[&str] = &["a", "b"]; // rr = 1.0
        let q2: &[&str] = &["x", "a"]; // rr = 0.5
        let got = mrr([(q1, &rel_a), (q2, &rel_a)]);
        assert!((got - 0.75).abs() < 1e-6);
        assert_eq!(mrr(std::iter::empty::<(&[&str], &HashSet<&str>)>()), 0.0);
    }

    #[test]
    fn recall_hand_computed() {
        let relevant: HashSet<&str> = ["a", "c"].into_iter().collect();
        assert_eq!(recall_at_k(&["a", "b", "c"], &relevant, 2), 0.5);
        assert_eq!(recall_at_k(&["a", "b", "c"], &relevant, 3), 1.0);
        let empty: HashSet<&str> = HashSet::new();
        assert_eq!(recall_at_k(&["a"], &empty, 3), 0.0);
    }
}
