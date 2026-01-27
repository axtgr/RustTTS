use tracing_subscriber::EnvFilter;

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .compact()
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("rusttts=debug".parse().unwrap()),
        )
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::info;

    #[test]
    fn test_tracing() {
        init_tracing();
        info!("Tracing test successful");
    }
}
