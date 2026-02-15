use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::Result;
use crate::pipeline::{PipelineConfig, PipelineOutput, execute_pipeline, execute_pipeline_into};
use crate::pool::get_or_create_pool;

/// Load and process a batch of images in parallel using a persistent rayon thread pool.
/// Short-circuits on the first error encountered.
pub fn load_batch(
    paths: &[PathBuf],
    config: &PipelineConfig,
    num_workers: usize,
) -> Result<Vec<PipelineOutput>> {
    let pool = get_or_create_pool(num_workers);

    pool.install(|| {
        paths
            .par_iter()
            .map(|p| execute_pipeline(p, config))
            .collect()
    })
}

/// Load a batch directly into a single contiguous f32 buffer [N, 3, H, W].
/// Requires `config.normalize` and known output dimensions (from crop).
/// Each worker writes directly into its slice — no intermediate Vec<f32> per image.
pub fn load_batch_contiguous(
    paths: &[PathBuf],
    config: &PipelineConfig,
    num_workers: usize,
    height: u32,
    width: u32,
) -> Result<Vec<f32>> {
    let pool = get_or_create_pool(num_workers);
    let n = paths.len();
    let plane_size = 3 * height as usize * width as usize;
    let mut batch_buf = vec![0.0f32; n * plane_size];

    // Split into non-overlapping mutable slices — one per image
    let chunks: Vec<&mut [f32]> = batch_buf.chunks_exact_mut(plane_size).collect();

    pool.install(|| {
        chunks
            .into_par_iter()
            .zip(paths.par_iter())
            .try_for_each(|(chunk, path)| execute_pipeline_into(path, config, chunk))
    })?;

    Ok(batch_buf)
}
