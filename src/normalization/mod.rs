//! Score normalization strategies.
//!
//! External sources emit scores on incompatible ranges (cosine similarity in
//! `[-1, 1]`, BM25 unbounded, probabilities in `[0, 1]`). Before linear
//! combination, each list's scores are normalized **per source list, never
//! across the fused pool**: pooling would let one source's range distort
//! another's distribution and would make a list's normalization depend on
//! unrelated lists.
//!
//! Edge-case policies are explicit and tested, not discovered:
//!
//! - [`MinMax`]: empty → no-op; single element or all-equal → all `1.0`
//!   (a source's only/uniform result is its best result).
//! - [`ZScore`]: zero variance → all `0.0` (every score sits on the mean).
//! - [`Softmax`]: the maximum is subtracted before exponentiation, so extreme
//!   magnitudes cannot overflow to infinity.
//!
//! NaN inputs follow the crate-wide GIGO policy (see the [`crate::core`]
//! module docs): they propagate, ordering stays deterministic.

/// Normalizes one source list's scores in place.
///
/// Implementations must be pure per-slice: no state carried across calls, no
/// dependence on other lists.
pub trait Normalizer {
    fn normalize(&self, scores: &mut [f32]);
}

/// Min-max normalization to `[0, 1]`.
///
/// `x → (x - min) / (max - min)`. Single-element and all-equal slices map to
/// `1.0` — the degenerate `0/0` case is defined away instead of yielding NaN.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinMax;

impl Normalizer for MinMax {
    fn normalize(&self, scores: &mut [f32]) {
        let Some(&first) = scores.first() else {
            return;
        };
        let (min, max) = scores.iter().skip(1).fold((first, first), |(lo, hi), &s| {
            (
                if s.total_cmp(&lo).is_lt() { s } else { lo },
                if s.total_cmp(&hi).is_gt() { s } else { hi },
            )
        });
        let range = max - min;
        if range == 0.0 {
            scores.fill(1.0);
        } else {
            for s in scores {
                *s = (*s - min) / range;
            }
        }
    }
}

/// Z-score (standard score) normalization.
///
/// `x → (x - mean) / stddev` (population standard deviation). Zero variance
/// maps every score to `0.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZScore;

impl Normalizer for ZScore {
    fn normalize(&self, scores: &mut [f32]) {
        if scores.is_empty() {
            return;
        }
        let n = scores.len() as f32;
        let mean = scores.iter().sum::<f32>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
        let stddev = variance.sqrt();
        if stddev == 0.0 {
            scores.fill(0.0);
        } else {
            for s in scores {
                *s = (*s - mean) / stddev;
            }
        }
    }
}

/// Softmax normalization: scores become a probability distribution.
///
/// The maximum is subtracted before exponentiation (`exp(x - max)`), the
/// standard trick that keeps extreme magnitudes finite: the largest exponent
/// is exactly `exp(0) = 1`, so the sum can never overflow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Softmax;

impl Normalizer for Softmax {
    fn normalize(&self, scores: &mut [f32]) {
        let Some(&first) = scores.first() else {
            return;
        };
        let max = scores.iter().skip(1).fold(
            first,
            |hi, &s| if s.total_cmp(&hi).is_gt() { s } else { hi },
        );
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores {
            *s /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_maps_to_unit_interval() {
        let mut s = [2.0, 6.0, 4.0];
        MinMax.normalize(&mut s);
        assert_eq!(s, [0.0, 1.0, 0.5]);
    }

    #[test]
    fn minmax_single_element_is_one() {
        let mut s = [5.0];
        MinMax.normalize(&mut s);
        assert_eq!(s, [1.0]);
    }

    #[test]
    fn minmax_all_equal_is_one() {
        let mut s = [3.0, 3.0, 3.0];
        MinMax.normalize(&mut s);
        assert_eq!(s, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn minmax_handles_negative_ranges() {
        let mut s = [-1.0, 1.0];
        MinMax.normalize(&mut s);
        assert_eq!(s, [0.0, 1.0]);
    }

    #[test]
    fn zscore_centers_and_scales() {
        let mut s = [1.0, 2.0, 3.0];
        ZScore.normalize(&mut s);
        // mean 2, population stddev sqrt(2/3)
        let sd = (2.0f32 / 3.0).sqrt();
        assert!((s[0] - (-1.0 / sd)).abs() < 1e-6);
        assert!((s[1]).abs() < 1e-6);
        assert!((s[2] - (1.0 / sd)).abs() < 1e-6);
    }

    #[test]
    fn zscore_zero_variance_is_zero() {
        let mut s = [3.0, 3.0, 3.0];
        ZScore.normalize(&mut s);
        assert_eq!(s, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_is_a_distribution_preserving_order() {
        let mut s = [1.0, 3.0, 2.0];
        Softmax.normalize(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(s[1] > s[2] && s[2] > s[0]);
    }

    #[test]
    fn softmax_extreme_magnitudes_stay_finite() {
        let mut s = [1e30, 1.0];
        Softmax.normalize(&mut s);
        assert!(s.iter().all(|v| v.is_finite()));
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert_eq!(s[1], 0.0);
    }

    #[test]
    fn empty_slices_are_noops() {
        let mut s: [f32; 0] = [];
        MinMax.normalize(&mut s);
        ZScore.normalize(&mut s);
        Softmax.normalize(&mut s);
    }
}
