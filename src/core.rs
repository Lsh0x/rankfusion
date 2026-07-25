//! Core data model: candidates, ranked/scored lists, and merge policy.
//!
//! # Design rationale
//!
//! ## Score status: two list types, no `Option<f32>`
//!
//! Rank fusion (RRF) is purely rank-based — it needs the *order* of a list,
//! never a score. Score fusion (linear combination) needs uniformly scored
//! lists. Modeling this as `score: Option<f32>` on every candidate would force
//! each algorithm to handle `None` defensively and could not guarantee that a
//! list is uniformly scored. Instead the invariant is carried by the type:
//!
//! - [`RankedList`] — an ordered list of [`Candidate`]s; rank is implicit
//!   (position 0 is rank 1). Input to rank-based fusion.
//! - [`ScoredList`] — an ordered list of [`Scored`] candidates; every entry
//!   has a score. Input to score-based fusion.
//!
//! A [`ScoredList`] converts to a [`RankedList`] for free (drop the scores —
//! see the [`From`] impl); the reverse is impossible by construction. Fusion
//! output is always `Vec<Scored<Id, M>>`: a fused score always exists.
//!
//! ## Metadata merge policy
//!
//! When the same candidate id appears in several input lists, fusion keeps
//! exactly one metadata value. The default policy is **first-wins**: the
//! metadata from the first list (in the order the lists are passed) that
//! contains the candidate is kept, subsequent occurrences are dropped. This is
//! deterministic and allocation-free. Custom behaviour plugs in through
//! [`MergePolicy`], which is implemented for any `Fn(&mut M, M)` closure.
//!
//! ## NaN and infinity policy
//!
//! `f32` is not `Ord`. Every internal score comparison in this crate uses
//! [`f32::total_cmp`], so sorting is deterministic even in the presence of
//! NaN. In `total_cmp` order positive NaN is the largest value and negative
//! NaN the smallest — so in a descending ranking, positive NaN (Rust's
//! `f32::NAN`) sorts *above* +inf. Input scores are *not* validated: feeding
//! NaN in yields NaN fused scores out (garbage in, garbage out), but ordering
//! remains total and deterministic, and no `Result` or NaN branch pollutes
//! the hot path. Compare with [`Scored::cmp_score_desc`].

/// A candidate produced by an external retrieval system.
///
/// `Id` is only ever used through `Eq + Hash + Clone` — `u64`, UUIDs,
/// `String`, or any custom type work. `Metadata` is an opaque payload carried
/// through the pipeline untouched; the library never interprets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Candidate<Id, Metadata = ()> {
    pub id: Id,
    pub metadata: Metadata,
}

impl<Id, Metadata> Candidate<Id, Metadata> {
    pub fn new(id: Id, metadata: Metadata) -> Self {
        Self { id, metadata }
    }
}

impl<Id> Candidate<Id> {
    /// A candidate with no metadata.
    pub fn bare(id: Id) -> Self {
        Self { id, metadata: () }
    }
}

/// A [`Candidate`] together with a score.
///
/// Produced by scored sources ([`ScoredList`]) and by every fusion strategy
/// (the fused score). Higher scores rank first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored<Id, Metadata = ()> {
    pub candidate: Candidate<Id, Metadata>,
    pub score: f32,
}

impl<Id, Metadata> Scored<Id, Metadata> {
    pub fn new(id: Id, score: f32, metadata: Metadata) -> Self {
        Self {
            candidate: Candidate::new(id, metadata),
            score,
        }
    }

    pub fn id(&self) -> &Id {
        &self.candidate.id
    }

    /// Total, deterministic descending-score ordering (see the module docs'
    /// NaN policy). Ties compare equal — callers needing a stable overall
    /// order should use a stable sort.
    pub fn cmp_score_desc(&self, other: &Self) -> std::cmp::Ordering {
        other.score.total_cmp(&self.score)
    }
}

/// An ordered candidate list; rank is implicit (position 0 = rank 1).
///
/// Lists may be empty, have any length, and share or omit candidates relative
/// to other lists — fusion handles all of it.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedList<Id, Metadata = ()> {
    pub items: Vec<Candidate<Id, Metadata>>,
}

impl<Id, Metadata> RankedList<Id, Metadata> {
    pub fn new(items: Vec<Candidate<Id, Metadata>>) -> Self {
        Self { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<Id, Metadata> Default for RankedList<Id, Metadata> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

impl<Id> FromIterator<Id> for RankedList<Id> {
    /// Build a metadata-less ranked list from ids, best first.
    fn from_iter<T: IntoIterator<Item = Id>>(iter: T) -> Self {
        Self::new(iter.into_iter().map(Candidate::bare).collect())
    }
}

/// An ordered, uniformly scored candidate list.
///
/// Order and scores are both provided by the producing system; this library
/// never re-sorts an input list (a source may deliberately rank against its
/// own scores).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredList<Id, Metadata = ()> {
    pub items: Vec<Scored<Id, Metadata>>,
}

impl<Id, Metadata> ScoredList<Id, Metadata> {
    pub fn new(items: Vec<Scored<Id, Metadata>>) -> Self {
        Self { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<Id, Metadata> Default for ScoredList<Id, Metadata> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

/// Dropping scores is free: a scored list is always a valid ranked list.
/// The reverse conversion does not exist, by design.
impl<Id, Metadata> From<ScoredList<Id, Metadata>> for RankedList<Id, Metadata> {
    fn from(scored: ScoredList<Id, Metadata>) -> Self {
        Self {
            items: scored.items.into_iter().map(|s| s.candidate).collect(),
        }
    }
}

/// Decides which metadata survives when a candidate appears in several lists.
///
/// `kept` is the metadata retained so far (from the earliest list containing
/// the candidate); `incoming` is the occurrence just encountered. The default
/// policy is [`FirstWins`]. Any `Fn(&mut M, M)` closure is also a policy:
///
/// ```
/// use rankfusion::MergePolicy;
///
/// // keep the maximum of two integer payloads
/// let max_wins = |kept: &mut u32, incoming: u32| *kept = (*kept).max(incoming);
/// let mut m = 3;
/// MergePolicy::merge(&max_wins, &mut m, 7);
/// assert_eq!(m, 7);
/// ```
pub trait MergePolicy<Metadata> {
    fn merge(&self, kept: &mut Metadata, incoming: Metadata);
}

/// Default merge policy: the first occurrence's metadata is kept, later
/// occurrences are dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FirstWins;

impl<Metadata> MergePolicy<Metadata> for FirstWins {
    fn merge(&self, _kept: &mut Metadata, _incoming: Metadata) {}
}

impl<Metadata, F> MergePolicy<Metadata> for F
where
    F: Fn(&mut Metadata, Metadata),
{
    fn merge(&self, kept: &mut Metadata, incoming: Metadata) {
        self(kept, incoming);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_ids() {
        let a = Candidate::bare(42u64);
        let b = Candidate::bare("doc-1".to_string());
        let c = Candidate::new(7u32, "payload");
        assert_eq!(a.id, 42);
        assert_eq!(b.id, "doc-1");
        assert_eq!(c.metadata, "payload");
    }

    #[test]
    fn empty_and_single_element_lists() {
        let empty: RankedList<u64> = RankedList::default();
        assert!(empty.is_empty());

        let single: RankedList<u64> = [1u64].into_iter().collect();
        assert_eq!(single.len(), 1);

        let scored = ScoredList::new(vec![Scored::new(1u64, 0.5, ())]);
        assert_eq!(scored.len(), 1);
    }

    #[test]
    fn scored_list_converts_to_ranked_list() {
        let scored = ScoredList::new(vec![Scored::new("a", 0.9, ()), Scored::new("b", 0.4, ())]);
        let ranked: RankedList<&str> = scored.into();
        let ids: Vec<&str> = ranked.items.iter().map(|c| c.id).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn first_wins_keeps_earliest_metadata() {
        let mut kept = "from-list-0";
        MergePolicy::merge(&FirstWins, &mut kept, "from-list-1");
        assert_eq!(kept, "from-list-0");
    }

    #[test]
    fn closure_merge_policy_overrides() {
        let concat = |kept: &mut String, incoming: String| {
            kept.push('+');
            kept.push_str(&incoming);
        };
        let mut kept = String::from("v1");
        MergePolicy::merge(&concat, &mut kept, String::from("v2"));
        assert_eq!(kept, "v1+v2");
    }

    #[test]
    fn nan_score_ordering_is_total_and_deterministic() {
        let mut items = [
            Scored::new("nan", f32::NAN, ()),
            Scored::new("low", -1.0, ()),
            Scored::new("high", 10.0, ()),
            Scored::new("neg-inf", f32::NEG_INFINITY, ()),
            Scored::new("pos-inf", f32::INFINITY, ()),
        ];
        items.sort_by(Scored::cmp_score_desc);
        let ids: Vec<&str> = items.iter().map(|s| *s.id()).collect();
        // total_cmp descending: +NaN > +inf > 10.0 > -1.0 > -inf
        assert_eq!(ids, ["nan", "pos-inf", "high", "low", "neg-inf"]);

        // deterministic: sorting again yields the same order
        let before = ids.clone();
        items.sort_by(Scored::cmp_score_desc);
        let after: Vec<&str> = items.iter().map(|s| *s.id()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn duplicate_candidate_across_lists_is_representable() {
        // Fusion (issue #2) consumes this shape; the model itself must allow
        // the same id in several lists with different metadata.
        let list_a: RankedList<u64, &str> = RankedList::new(vec![
            Candidate::new(1, "meta-a"),
            Candidate::new(2, "meta-a"),
        ]);
        let list_b: RankedList<u64, &str> = RankedList::new(vec![Candidate::new(1, "meta-b")]);
        assert_eq!(list_a.items[0].id, list_b.items[0].id);
        assert_ne!(list_a.items[0].metadata, list_b.items[0].metadata);
    }
}
