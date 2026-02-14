use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::Result;
use crate::pipeline::{PipelineConfig, PipelineOutput, execute_pipeline};

/// Load and process a batch of images in parallel using a dedicated rayon thread pool.
/// Short-circuits on the first error encountered.
pub fn load_batch(
    paths: &[PathBuf],
    config: &PipelineConfig,
    num_workers: usize,
) -> Result<Vec<PipelineOutput>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()
        .map_err(|e| crate::error::TensorImageError::InvalidParam(e.to_string()))?;

    pool.install(|| {
        paths
            .par_iter()
            .map(|p| execute_pipeline(p, config))
            .collect()
    })
}
