#!/bin/sh
# Publish the kredo crates to crates.io in dependency order.
# Requires: cargo login <token>
set -eu

crates="kredo-api kredo-decision kredo-lang kredo-registry kredo-runner kredo-server kredo"

for c in $crates; do
    echo "== publishing $c"
    cargo publish -p "$c"
    # crates.io index propagation
    sleep 8
done
echo "all crates published"
