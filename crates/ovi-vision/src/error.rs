/// Errors that can occur when using ovi-vision.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Video capture error: {0}")]
    VideoCapture(String),

    #[error("Gallery error: {0}")]
    Gallery(String),

    #[error("CXX bridge error: {0}")]
    Cxx(#[from] cxx::Exception),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = Error::ModelLoad("bad.xml".into());
        assert!(e.to_string().contains("bad.xml"));
        assert!(e.to_string().contains("model"));

        let e = Error::Inference("timeout".into());
        assert!(e.to_string().contains("timeout"));

        let e = Error::VideoCapture("not found".into());
        assert!(e.to_string().contains("not found"));

        let e = Error::Gallery("empty".into());
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_all_error_variants_display() {
        let cases: Vec<(Error, &str)> = vec![
            (Error::ModelLoad("path".into()), "Failed to load model: path"),
            (Error::Inference("oom".into()), "Inference failed: oom"),
            (Error::VideoCapture("no cam".into()), "Video capture error: no cam"),
            (Error::Gallery("missing".into()), "Gallery error: missing"),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn test_error_debug() {
        let e = Error::ModelLoad("test".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("ModelLoad"));
    }
}
