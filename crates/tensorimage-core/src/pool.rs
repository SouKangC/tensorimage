use std::sync::OnceLock;

static GLOBAL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

pub fn get_or_create_pool(num_workers: usize) -> &'static rayon::ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .expect("Failed to create thread pool")
    })
}
