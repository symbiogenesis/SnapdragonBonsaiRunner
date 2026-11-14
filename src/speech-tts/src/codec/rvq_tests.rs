//! Literal nearest-centroid answers and refusal/rollback controls for reference RVQ.

use std::io;

use super::{RvqCodeAppend, RvqEncodeSide};

fn residual_side() -> io::Result<RvqEncodeSide> {
    RvqEncodeSide::new(
        5,
        2,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![
            vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0],
            vec![0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        ],
    )
}

#[test]
fn rvq_reference_literal_residual_layers_keep_projection_ties_and_codebook_major_order()
-> io::Result<()> {
    let side = residual_side()?;
    let mut codes = vec![91, -7];
    // Projection discards the third coordinate. First-layer residuals are
    // [1, 1], [-1, 1], [0, 0], [0, -1]. Their second-layer nearest entries
    // are 1, 2, 0, and the equal-distance 0/3/4 tie's first entry, 0.
    // The last first-layer frame ties entries 0 and 4 across SIMD/tail paths.
    side.encode(
        &[
            3.0, 1.0, 99.0, -1.0, 3.0, -99.0, 0.0, 0.0, 8.0, 0.0, -1.0, 9.0,
        ],
        4,
        &mut codes,
    )?;
    assert_eq!(codes, [91, -7, 1, 2, 0, 0, 1, 2, 0, 0]);
    assert_eq!(
        side.sqsum,
        [vec![0.0, 4.0, 4.0, 4.0, 4.0], vec![0.0, 2.0, 2.0, 2.0, 2.0]]
    );
    Ok(())
}

#[test]
fn rvq_reference_refuses_nonfinite_weights_norms_projections_and_candidate_scores() -> io::Result<()>
{
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(RvqEncodeSide::new(1, 1, 1, vec![value], vec![vec![0.0]]).is_err());
        assert!(RvqEncodeSide::new(1, 1, 1, vec![1.0], vec![vec![value]]).is_err());
    }
    // Finite model values can overflow centroid norms, projection sums, or the
    // -2*dot + norm score; no such case may silently select centroid zero.
    assert!(RvqEncodeSide::new(1, 1, 1, vec![1.0], vec![vec![f32::MAX]]).is_err());
    let projection = RvqEncodeSide::new(1, 1, 1, vec![2.0], vec![vec![0.0]])?;
    let score = RvqEncodeSide::new(1, 1, 1, vec![1.0], vec![vec![1.0]])?;
    for side in [projection, score] {
        let mut codes = vec![12, -9];
        assert!(side.encode(&[f32::MAX], 1, &mut codes).is_err());
        assert_eq!(codes, [12, -9]);
    }
    let side = residual_side()?;
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut codes = vec![12, -9];
        assert!(side.encode(&[value, 0.0, 0.0], 1, &mut codes).is_err());
        assert_eq!(codes, [12, -9]);
    }
    Ok(())
}

#[test]
fn rvq_reference_late_score_failure_and_unwind_preserve_the_original_output() -> io::Result<()> {
    for bad_entry in [0, 4] {
        let mut side = residual_side()?;
        // The complete first layer has already appended two codes before this
        // invalid second-layer score reaches the grouped or tail argmin path.
        side.sqsum[1][bad_entry] = f32::NAN;
        let mut codes = vec![19, -4];
        assert!(
            side.encode(&[3.0, 1.0, 0.0, -1.0, 3.0, 0.0], 2, &mut codes)
                .is_err()
        );
        assert_eq!(codes, [19, -4]);
    }
    let mut codes = vec![19, -4];
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let append = RvqCodeAppend::new(&mut codes);
        append.codes.extend([3, 8]);
        std::panic::resume_unwind(Box::new("injected reference-code publication fault"));
    }));
    assert!(outcome.is_err());
    assert_eq!(codes, [19, -4]);
    Ok(())
}

#[test]
fn rvq_reference_invalid_mutable_geometry_is_rejected_without_panics_or_output() -> io::Result<()> {
    let original = residual_side()?;
    for mutation in 0..9 {
        let mut side = original.clone();
        match mutation {
            0 => side.num_layers = 0,
            1 => side.codebook_size = 0,
            2 => side.codebook_dim = usize::MAX,
            3 => side.hidden_size = usize::MAX,
            4 => {
                let _ = side.input_proj.pop();
            }
            5 => {
                let _ = side.codebooks.pop();
            }
            6 => {
                let _ = side.codebooks[0].pop();
            }
            7 => {
                let _ = side.sqsum.pop();
            }
            _ => {
                let _ = side.sqsum[0].pop();
            }
        }
        let mut codes = vec![72];
        assert!(side.encode(&[1.0, 2.0, 3.0], 1, &mut codes).is_err());
        assert_eq!(codes, [72]);
    }
    for frames in [0, usize::MAX] {
        let mut codes = vec![72];
        assert!(original.encode(&[], frames, &mut codes).is_err());
        assert_eq!(codes, [72]);
    }
    Ok(())
}
