# Golden fixtures

Each fixture pins a state, a question set and expected answer properties for
a pulled model. Tests that require a pulled model are `#[ignore]`d by default;
run them explicitly after `kredo pull`:

    kredo pull kredo:en
    cargo test -p kredo-runner -- --ignored

CI downloads and verifies the pinned models, then runs the ignored tests.
