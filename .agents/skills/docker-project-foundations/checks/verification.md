# Verification Checklist

Use this checklist to verify that generated Docker project setup follows the skill's guidance.

## Required files

- [ ] `.dockerignore` exists at the project root (or at each build context root in a multi-service setup).
- [ ] `Dockerfile` exists at the project root (or at each build context root).
- [ ] `compose.yaml` exists at the project root.

## .dockerignore

- [ ] Excludes `.git` directory.
- [ ] Excludes dependency caches (`node_modules/`, `__pycache__/`, `.venv/`, `vendor/`).
- [ ] Excludes IDE/editor configs (`.vscode/`, `.idea/`).
- [ ] Excludes secret files (`.env`, `*.pem`, `*.key`).
- [ ] Excludes root and nested npm credential files with `**/.npmrc`.
- [ ] Does not exclude files that the build actually needs (source code, dependency manifests).

## Dockerfile

- [ ] Base image uses a specific version tag, not `latest`.
- [ ] Base image uses a minimal variant (`-slim` or `-alpine`) where available.
- [ ] Dependency manifests are copied and installed before source code (layer caching).
- [ ] A non-root `USER` is set before `CMD`/`ENTRYPOINT`.
- [ ] No secrets or credentials are hardcoded (`ENV SECRET=...`, `ARG PASSWORD=...`).
- [ ] Neither `COPY` nor `ADD` includes `.npmrc`; both npm installation steps use an optional BuildKit secret mount instead.
- [ ] Public-package builds work without `.npmrc`; private-registry builds receive it through `--secret id=npmrc,src=<config-path>`. Never test with real credentials in image layers or logs.
- [ ] Multi-stage build is used when a build step exists (compile, bundle, transpile).
- [ ] Production stage does not contain dev tools, test frameworks, or build toolchains.

## compose.yaml

- [ ] File is named `compose.yaml`, not `docker-compose.yml`.
- [ ] Infrastructure dependencies (databases, caches, queues) are defined as Compose services, not expected to be installed on the host.
- [ ] `depends_on` uses `condition: service_healthy` for services that need readiness.
- [ ] Infrastructure services have `healthcheck` definitions.
- [ ] Persistent data uses named volumes, not bind mounts.
- [ ] Application source code uses bind mounts for development live-reload.
- [ ] Application and datastore credentials use Compose interpolation rather than literal values, including passwords embedded in connection URLs.
- [ ] Application ports bind to loopback by default; widening to other interfaces is an explicit development choice.
- [ ] Unauthenticated datastores are not published to the host and remain reachable only on the Compose network.
- [ ] Datastore ports needed by local host tools bind to loopback only.
- [ ] Development-only credential fallbacks are clearly labeled, use Compose interpolation, and document a `.env` override.
- [ ] No host-level install instructions (`brew install`, `apt install`) for services that should be containerized.

## Development vs production

- [ ] Development configuration uses bind mounts for source code.
- [ ] Production configuration does not mount source code.
- [ ] A single `Dockerfile` supports both via build stages or build arguments when feasible.

## Validation script

Run the bundled script from the project root before the broader smoke tests:

```bash
bash scripts/verify-setup.sh [--help]
```

It checks `.dockerignore`, `Dockerfile`, and `compose.yaml`, then validates the Compose configuration. Exit status is `0` on success or help, `1` for missing files or invalid Compose configuration, and `2` for invalid arguments.

## Validation commands

Run these to smoke-test the generated setup:

```bash
# Verify compose file is syntactically valid
docker compose config --quiet

# Verify the Dockerfile builds successfully
docker compose build

# Verify services start and become healthy
docker compose up -d
docker compose ps   # All services should show "healthy" or "running"

# Clean up
docker compose down -v
```
