# Contributing

Thanks for considering a contribution to kredo.

## Development

```
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

All three must pass; CI enforces them. Model work additionally requires the
parity gate (see below).

## Pull requests

- Keep changes focused; one logical change per PR.
- Update `CHANGELOG.md` under the unreleased heading.
- New endpoints need integration tests (`crates/kredo-server/tests`).
- New inference behavior needs a golden fixture (see below).

## Model changes

Models are built by the `convert/` pipeline and are never hand-edited:

1. Extend `convert/kredo_convert/` (question set, dataset loader).
2. Train and export.
3. Run the parity gate: ONNX must match PyTorch within 1e-4 on the eval
   distribution. This is a hard gate — a model that fails parity is not
   packaged.
4. Package (writes artifacts + digests), wire the manifest in
   `crates/kredo-registry/src/library.rs`, and re-run the ignored golden
   tests:

   ```
   kredo pull kredo:<tag>
   KREDO_MODELS=... cargo test -p kredo-runner -- --ignored
   ```

Third-party checkpoints are pulled from their authors at pinned revisions
with sha256 verification; never re-host or commit third-party weights.

## Reporting security issues

See docs/security.md for the threat model. For vulnerabilities, please open
a private security advisory rather than a public issue.
