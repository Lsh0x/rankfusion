//! Composing business rules on top of fusion with a reranking pipeline.
//!
//! Run with: `cargo run --example pipeline`

use rankfusion::{Candidate, Pipeline, RankedList, Rrf, Scored, TopK};

/// What an application typically carries alongside an id.
#[derive(Debug, Clone, Copy)]
struct Doc {
    age_days: u32,
    promoted: bool,
}

fn main() {
    let vector_search: RankedList<&str, Doc> = RankedList::new(vec![
        Candidate::new(
            "doc-7",
            Doc {
                age_days: 400,
                promoted: false,
            },
        ),
        Candidate::new(
            "doc-3",
            Doc {
                age_days: 2,
                promoted: false,
            },
        ),
        Candidate::new(
            "doc-1",
            Doc {
                age_days: 30,
                promoted: true,
            },
        ),
    ]);
    let keyword_search: RankedList<&str, Doc> = RankedList::new(vec![Candidate::new(
        "doc-7",
        Doc {
            age_days: 400,
            promoted: false,
        },
    )]);

    // Stages are plain closures over the fused candidates. They run in
    // declaration order, each seeing what the previous one produced.
    let freshness = |candidates: &mut Vec<Scored<&str, Doc>>| {
        for candidate in candidates.iter_mut() {
            if candidate.candidate.metadata.age_days < 7 {
                candidate.score *= 1.5;
            }
        }
        candidates.sort_by(Scored::cmp_score_desc);
    };

    let promotions = |candidates: &mut Vec<Scored<&str, Doc>>| {
        for candidate in candidates.iter_mut() {
            if candidate.candidate.metadata.promoted {
                candidate.score += 0.01;
            }
        }
        candidates.sort_by(Scored::cmp_score_desc);
    };

    let results = Pipeline::new(Rrf::default())
        .reranker(freshness)
        .reranker(promotions)
        // truncation goes last, once the order has settled
        .reranker(TopK::new(3))
        .rank(vec![vector_search, keyword_search])
        .expect("infallible fusion");

    println!("rank  id      score    age  promoted");
    for (position, result) in results.iter().enumerate() {
        let doc = result.candidate.metadata;
        println!(
            "{:>4}  {:<6}  {:.5}  {:>3}  {}",
            position + 1,
            result.id(),
            result.score,
            doc.age_days,
            doc.promoted,
        );
    }
}
