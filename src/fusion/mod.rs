//! Fusion strategies: combine multiple input lists into one scored ranking.
//!
//! All strategies share the same accumulation core: a single pass over every
//! input candidate into a `HashMap<Id, _>` accumulator, then one final sort —
//! `O(total_candidates + n log n)`, no quadratic scans. Ties on the fused
//! score are broken by first-seen order (list order, then rank), which makes
//! the output deterministic regardless of the hash map's iteration order or
//! hasher choice.
//!
//! Fusion takes ownership of its input lists so candidate metadata moves into
//! the result without cloning; only `Id: Clone` is required (one clone per
//! distinct candidate, used as the accumulator key).

mod linear;
mod rrf;

pub use linear::LinearFusion;
pub use rrf::{Rrf, WeightedRrf};

use std::collections::hash_map::Entry;
use std::hash::Hash;

/// Accumulator map: `std::collections::HashMap` by default, `ahash::AHashMap`
/// behind the `ahash` feature. The hasher never affects ranking output — ties
/// break by first-seen order, not iteration order.
#[cfg(feature = "ahash")]
pub(crate) type FxMap<K, V> = ahash::AHashMap<K, V>;
#[cfg(not(feature = "ahash"))]
pub(crate) type FxMap<K, V> = std::collections::HashMap<K, V>;

use crate::core::{Candidate, MergePolicy, RankedList, Scored};
use crate::explain::{Explained, SourceContribution};

/// Errors produced by fallible fusion strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FusionError {
    /// The number of weights does not match the number of input lists.
    WeightCountMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightCountMismatch { expected, got } => write!(
                f,
                "weight count mismatch: {expected} weights for {got} lists"
            ),
        }
    }
}

impl std::error::Error for FusionError {}

/// A fusion strategy usable in a [`crate::reranking::Pipeline`].
///
/// The input list type is an associated type: rank-based strategies
/// ([`Rrf`], [`WeightedRrf`]) consume [`RankedList`]s, score-based ones
/// ([`LinearFusion`]) consume [`crate::ScoredList`]s — the compiler enforces
/// the match, mixing is impossible. The return is uniformly `Result` so
/// pipelines have a single signature; infallible strategies always return
/// `Ok`.
pub trait Fusion<Id, Metadata> {
    /// The input list type this strategy consumes.
    type Input;

    fn fuse(&self, lists: Vec<Self::Input>) -> Result<Vec<Scored<Id, Metadata>>, FusionError>;

    /// [`Fusion::fuse`] with per-source contribution tracing — see
    /// [`crate::explain`]. Runs a separate accumulation path: `fuse` itself
    /// pays no cost for this method's existence.
    fn fuse_explained(
        &self,
        lists: Vec<Self::Input>,
    ) -> Result<Vec<Explained<Id, Metadata>>, FusionError>;
}

struct Acc<Id, Metadata> {
    scored: Scored<Id, Metadata>,
    first_seen: usize,
}

/// Shared accumulation core for every fusion strategy: merge `(candidate,
/// contribution)` pairs into a hash map (summing contributions, merging
/// metadata via `policy`), then sort by fused score descending. Ties break by
/// first-seen order, so the output is deterministic regardless of the map's
/// iteration order or hasher.
pub(crate) fn accumulate_contributions<Id, Metadata, P>(
    contributions: impl IntoIterator<Item = (Candidate<Id, Metadata>, f32)>,
    policy: &P,
) -> Vec<Scored<Id, Metadata>>
where
    Id: Eq + Hash + Clone,
    P: MergePolicy<Metadata>,
{
    let mut acc: FxMap<Id, Acc<Id, Metadata>> = FxMap::default();
    let mut first_seen = 0usize;

    for (candidate, contribution) in contributions {
        match acc.entry(candidate.id.clone()) {
            Entry::Occupied(mut entry) => {
                let slot = entry.get_mut();
                slot.scored.score += contribution;
                policy.merge(&mut slot.scored.candidate.metadata, candidate.metadata);
            }
            Entry::Vacant(entry) => {
                entry.insert(Acc {
                    scored: Scored {
                        candidate,
                        score: contribution,
                    },
                    first_seen,
                });
                first_seen += 1;
            }
        }
    }

    let mut out: Vec<Acc<Id, Metadata>> = acc.into_values().collect();
    out.sort_unstable_by(|a, b| {
        b.scored
            .score
            .total_cmp(&a.scored.score)
            .then(a.first_seen.cmp(&b.first_seen))
    });
    out.into_iter().map(|a| a.scored).collect()
}

struct AccExplained<Id, Metadata> {
    explained: Explained<Id, Metadata>,
    first_seen: usize,
}

/// Explained twin of [`accumulate_contributions`]: same map, same sort, same
/// first-seen tie-break, but each entry carries its [`SourceContribution`].
/// Kept as a separate path so the plain accumulator stays untouched.
pub(crate) fn accumulate_explained<Id, Metadata, P>(
    entries: impl IntoIterator<Item = (Candidate<Id, Metadata>, SourceContribution)>,
    policy: &P,
) -> Vec<Explained<Id, Metadata>>
where
    Id: Eq + Hash + Clone,
    P: MergePolicy<Metadata>,
{
    let mut acc: FxMap<Id, AccExplained<Id, Metadata>> = FxMap::default();
    let mut first_seen = 0usize;

    for (candidate, contribution) in entries {
        match acc.entry(candidate.id.clone()) {
            Entry::Occupied(mut entry) => {
                let slot = entry.get_mut();
                slot.explained.scored.score += contribution.partial_score;
                slot.explained.contributions.push(contribution);
                policy.merge(
                    &mut slot.explained.scored.candidate.metadata,
                    candidate.metadata,
                );
            }
            Entry::Vacant(entry) => {
                entry.insert(AccExplained {
                    explained: Explained {
                        scored: Scored {
                            candidate,
                            score: contribution.partial_score,
                        },
                        contributions: vec![contribution],
                    },
                    first_seen,
                });
                first_seen += 1;
            }
        }
    }

    let mut out: Vec<AccExplained<Id, Metadata>> = acc.into_values().collect();
    out.sort_unstable_by(|a, b| {
        b.explained
            .scored
            .score
            .total_cmp(&a.explained.scored.score)
            .then(a.first_seen.cmp(&b.first_seen))
    });
    out.into_iter().map(|a| a.explained).collect()
}

/// Explained twin of [`accumulate_rrf`].
pub(crate) fn explained_rrf<Id, Metadata, P>(
    k: f32,
    lists: Vec<RankedList<Id, Metadata>>,
    weights: Option<&[f32]>,
    policy: &P,
) -> Vec<Explained<Id, Metadata>>
where
    Id: Eq + Hash + Clone,
    P: MergePolicy<Metadata>,
{
    let entries = lists
        .into_iter()
        .enumerate()
        .flat_map(|(list_index, list)| {
            let weight = weights.map_or(1.0, |w| w[list_index]);
            list.items
                .into_iter()
                .enumerate()
                .map(move |(position, candidate)| {
                    let rank = position + 1;
                    (
                        candidate,
                        SourceContribution {
                            list_index,
                            rank,
                            partial_score: weight / (k + rank as f32),
                        },
                    )
                })
        });
    accumulate_explained(entries, policy)
}

/// RRF accumulation: `score(c) = Σ weight_i / (k + rank_i(c))`, rank starting
/// at 1. `weights`, when present, is already validated to match `lists` in
/// length.
pub(crate) fn accumulate_rrf<Id, Metadata, P>(
    k: f32,
    lists: Vec<RankedList<Id, Metadata>>,
    weights: Option<&[f32]>,
    policy: &P,
) -> Vec<Scored<Id, Metadata>>
where
    Id: Eq + Hash + Clone,
    P: MergePolicy<Metadata>,
{
    let contributions = lists
        .into_iter()
        .enumerate()
        .flat_map(|(list_index, list)| {
            let weight = weights.map_or(1.0, |w| w[list_index]);
            list.items
                .into_iter()
                .enumerate()
                .map(move |(position, candidate)| {
                    let rank = position + 1;
                    (candidate, weight / (k + rank as f32))
                })
        });
    accumulate_contributions(contributions, policy)
}
