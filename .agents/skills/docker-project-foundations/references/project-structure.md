# Docker Project Structure Reference

## Standard single-service layout

```
project-root/
  .dockerignore
  Dockerfile
  compose.yaml
  src/
  ...
```

For most projects, all Docker files live at the project root. This is the simplest and most conventional layout.

## Multi-service layout

When a repository contains multiple independently built services (e.g., a monorepo with `frontend/` and `backend/`):

```
project-root/
  compose.yaml
  frontend/
    .dockerignore
    Dockerfile
    src/
  backend/
    .dockerignore
    Dockerfile
    src/
```

Each service gets its own `Dockerfile` and `.dockerignore` at the root of its build context. The `compose.yaml` stays at the repository root and references each service's build context:

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
```

## File naming conventions

| File | Correct name | Deprecated/incorrect alternatives |
|------|-------------|----------------------------------|
| Compose file | `compose.yaml` | `docker-compose.yml`, `docker-compose.yaml` |
| Dockerfile | `Dockerfile` | `dockerfile`, `Dockerfile.dev` (use stages instead) |
| Ignore file | `.dockerignore` | — |

## Build context considerations

The Docker build context is the directory tree sent to the Docker daemon during a build. Key rules:

- The `.dockerignore` file controls what is excluded from the build context.
- A smaller build context means faster builds. Aggressively exclude anything the build does not need.
- The build context root is set by the `context` field in `compose.yaml` or by the path argument to `docker build`.
- Files outside the build context cannot be referenced in a `Dockerfile` (no `COPY ../something`).

## Compose file organization

### Environment variables

Prefer inline `environment:` blocks for small numbers of variables. Use `env_file:` for larger configurations, but never commit files containing real secrets.

```yaml
services:
  app:
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/myapp
      - REDIS_URL=redis://cache:6379
```

### Volumes

- **Named volumes** for data that must survive container restarts (database storage):

```yaml
volumes:
  db-data:

services:
  db:
    volumes:
      - db-data:/var/lib/postgresql/data
```

- **Bind mounts** for source code during development:

```yaml
services:
  app:
    volumes:
      - ./src:/app/src
```

### Networks

For most single-project development setups, the default Compose network is sufficient. Do not create custom networks unless services need isolation from each other.

### Profiles

Use Compose profiles to group optional services (e.g., monitoring, debug tools) that are not needed in every development session:

```yaml
services:
  prometheus:
    profiles:
      - monitoring
    image: prom/prometheus:v3.3
```

Start with `docker compose --profile monitoring up` when needed.

## Secrets and credentials

- Never bake secrets into images (no `ENV SECRET_KEY=...` in a `Dockerfile`).
- Use environment variables or Docker secrets for runtime credentials.
- Add secret files (`.env`, `*.pem`, `credentials.json`) to `.dockerignore` and `.gitignore`.
- For development, use `env_file:` in Compose pointing to a `.env` file that is gitignored.
