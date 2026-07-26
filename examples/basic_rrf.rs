//! Fusing two ranked sources with Reciprocal Rank Fusion.
//!
//! Run with: `cargo run --example basic_rrf`

use rankfusion::{RankedList, Rrf};

fn main() {
    // Two retrieval systems answered the same query. Neither knows about the
    // other, and their scores are not comparable — only their order is.
    let vector_search: RankedList<&str> = ["doc-7", "doc-3", "doc-1"].into_iter().collect();
    let keyword_search: RankedList<&str> = ["doc-3", "doc-9", "doc-7"].into_iter().collect();

    let fused = Rrf::default().fuse(vec![vector_search, keyword_search]);

    println!("rank  id      score");
    for (position, result) in fused.iter().enumerate() {
        println!(
            "{:>4}  {:<6}  {:.5}",
            position + 1,
            result.id(),
            result.score
        );
    }

    // doc-3 wins: it is the only document ranked highly by *both* sources.
    println!("\ntop result: {}", fused[0].id());
}
