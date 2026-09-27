# kredo — production image (CPU).
# The ort download-binaries feature statically links ONNX Runtime, so the
# runtime image needs no native deps beyond the base.
FROM rust:1-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release -p kredo

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --create-home --shell /usr/sbin/nologin kredo
USER kredo
WORKDIR /home/kredo
ENV KREDO_HOST=0.0.0.0:21435 \
    KREDO_MODELS=/data/models \
    KREDO_LOG_FORMAT=json
COPY --from=build /src/target/release/kredo /usr/local/bin/kredo
VOLUME /data/models
EXPOSE 21435
HEALTHCHECK --interval=30s --timeout=3s CMD kredo status || exit 1
ENTRYPOINT ["kredo"]
CMD ["serve"]
