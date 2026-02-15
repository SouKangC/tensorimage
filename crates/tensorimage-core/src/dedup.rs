use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::Result;
use crate::phash::{HashAlgorithm, hamming_distance, hash_file};
use crate::pool::get_or_create_pool;

/// Result of deduplication.
pub struct DedupResult {
    /// Indices of images to keep (first occurrence of each group).
    pub keep_indices: Vec<usize>,
    /// Groups of near-duplicate indices (each group has ≥2 members).
    pub duplicate_groups: Vec<Vec<usize>>,
    /// Hash per input image (in original order).
    pub hashes: Vec<u64>,
}

/// Deduplicate image files by perceptual hash.
///
/// 1. Computes hashes in parallel via rayon.
/// 2. Greedy groups by Hamming distance threshold.
///
/// Default threshold: 0 for DHash (exact), 10 for PHash.
pub fn deduplicate_paths(
    paths: &[PathBuf],
    algorithm: HashAlgorithm,
    threshold: u32,
    num_workers: usize,
) -> Result<DedupResult> {
    let pool = get_or_create_pool(num_workers);

    // Parallel hash computation
    let hashes: Result<Vec<u64>> =
        pool.install(|| paths.par_iter().map(|p| hash_file(p, algorithm)).collect());
    let hashes = hashes?;

    Ok(deduplicate_hashes(&hashes, threshold))
}

/// Deduplicate from pre-computed hashes using greedy grouping.
///
/// O(n²) pairwise comparison — acceptable for <100K images.
pub fn deduplicate_hashes(hashes: &[u64], threshold: u32) -> DedupResult {
    let n = hashes.len();
    // Track which group each image belongs to (-1 = not yet assigned)
    let mut group_of: Vec<i64> = vec![-1; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..n {
        if group_of[i] >= 0 {
            continue; // already assigned
        }
        // Start a new group with this image as representative
        let group_id = groups.len() as i64;
        let mut group = vec![i];
        group_of[i] = group_id;

        // Check remaining unassigned images
        for j in (i + 1)..n {
            if group_of[j] >= 0 {
                continue;
            }
            if hamming_distance(hashes[i], hashes[j]) <= threshold {
                group_of[j] = group_id;
                group.push(j);
            }
        }
        groups.push(group);
    }

    // keep_indices = first member of each group
    let keep_indices: Vec<usize> = groups.iter().map(|g| g[0]).collect();
    // duplicate_groups = only groups with ≥2 members
    let duplicate_groups: Vec<Vec<usize>> = groups.into_iter().filter(|g| g.len() >= 2).collect();

    DedupResult {
        keep_indices,
        duplicate_groups,
        hashes: hashes.to_vec(),
    }
}
