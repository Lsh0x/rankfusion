//! Answering "why is this result first?" with per-source contribution tracing.
//!
//! Run with: `cargo run --example explain`

use rankfusion::{RankedList, WeightedRrf};

fn main() {
    let sources = ["vector", "keyword"];
    let vector_search: RankedList<&str> = ["doc-7", "doc-3", "doc-1"].into_iter().collect();
    let keyword_search: RankedList<&str> = ["doc-3", "doc-9", "doc-7"].into_iter().collect();

    // The keyword source is trusted twice as much as the vector one.
    let fusion = WeightedRrf::new(60.0, vec![1.0, 2.0]);
    let explained = fusion
        .fuse_explained(vec![vector_search, keyword_search])
        .expect("one weight per input list");

    for result in &explained {
        println!("{} — fused score {:.5}", result.id(), result.score());
        for contribution in &result.contributions {
            println!(
                "    {:<8} rank {} → {:.5} ({:.0}% of the total)",
                sources[contribution.list_index],
                contribution.rank,
                contribution.partial_score,
                100.0 * contribution.partial_score / result.score(),
            );
        }
    }

    // The partial scores always add back up to the fused score, whatever the
    // fusion strategy — that is the invariant explainability rests on.
    let top = &explained[0];
    let sum: f32 = top.contributions.iter().map(|c| c.partial_score).sum();
    println!("\ncheck: {:.5} == {:.5}", sum, top.score());
}
