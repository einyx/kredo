//! Embedded decision playground: a single-page UI served at `/ui`.
//! No runtime network dependencies — fonts, styles and logic ship inside
//! the binary.

use axum::response::{Html, IntoResponse, Response};

const INDEX: &str = include_str!("ui/index.html");

const FONT_SG700: &[u8] = include_bytes!("ui/fonts/spacegrotesk-700.woff2");
const FONT_SG500: &[u8] = include_bytes!("ui/fonts/spacegrotesk-500.woff2");
const FONT_GEIST400: &[u8] = include_bytes!("ui/fonts/geist-400.woff2");
const FONT_GEIST500: &[u8] = include_bytes!("ui/fonts/geist-500.woff2");

async fn index() -> Html<&'static str> {
    Html(INDEX)
}

fn font(bytes: &'static [u8]) -> Response {
    (
        [
            ("content-type", "font/woff2"),
            ("cache-control", "public, max-age=604800, immutable"),
        ],
        bytes,
    )
        .into_response()
}

async fn font_sg700() -> Response {
    font(FONT_SG700)
}
async fn font_sg500() -> Response {
    font(FONT_SG500)
}
async fn font_geist400() -> Response {
    font(FONT_GEIST400)
}
async fn font_geist500() -> Response {
    font(FONT_GEIST500)
}

pub fn routes() -> axum::Router {
    use axum::routing::get;
    axum::Router::new()
        .route("/ui", get(index))
        .route("/ui/fonts/spacegrotesk-700.woff2", get(font_sg700))
        .route("/ui/fonts/spacegrotesk-500.woff2", get(font_sg500))
        .route("/ui/fonts/geist-400.woff2", get(font_geist400))
        .route("/ui/fonts/geist-500.woff2", get(font_geist500))
}
