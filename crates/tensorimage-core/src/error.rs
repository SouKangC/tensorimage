use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorImageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to decode image: {0}")]
    Decode(String),

    #[error("Unsupported image format: {0}")]
    UnsupportedFormat(String),

    #[error("Resize error: {0}")]
    Resize(String),

    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

pub type Result<T> = std::result::Result<T, TensorImageError>;
