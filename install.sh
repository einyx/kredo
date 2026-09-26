#!/bin/sh
# kredo installer: curl -fsSL https://example.com/install.sh | sh
set -eu

REPO="einyx/kredo"
INSTALL_DIR="${KREDO_INSTALL:-$HOME/.local/bin}"

main() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os-$arch" in
        Darwin-arm64) target=aarch64-apple-darwin ;;
        Darwin-x86_64) target=x86_64-apple-darwin ;;
        Linux-x86_64) target=x86_64-unknown-linux-gnu ;;
        Linux-aarch64) target=aarch64-unknown-linux-gnu ;;
        *)
            echo "unsupported platform: $os-$arch" >&2
            exit 1
            ;;
    esac

    version="${KREDO_VERSION:-latest}"
    if [ "$version" = "latest" ]; then
        version=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
    fi
    [ -n "$version" ] || { echo "could not determine latest release" >&2; exit 1; }

    url="https://github.com/$REPO/releases/download/$version/kredo-$version-$target.tar.gz"
    tmp=$(mktemp -d)
    echo "downloading $url"
    curl -fsSL "$url" | tar xz -C "$tmp"

    mkdir -p "$INSTALL_DIR"
    mv "$tmp"/kredo-*/kredo "$INSTALL_DIR/kredo"
    chmod +x "$INSTALL_DIR/kredo"
    echo "installed kredo $version -> $INSTALL_DIR/kredo"
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) ;;
        *) echo "note: $INSTALL_DIR is not on your PATH" ;;
    esac
}

main "$@"
