#!/usr/bin/env bash
# Verify Docker project setup. Run from the project root.
# Usage: bash scripts/verify-setup.sh [--help]
set -euo pipefail

usage() {
    echo "Usage: bash scripts/verify-setup.sh [--help]"
    echo "Checks: .dockerignore, Dockerfile, and compose.yaml exist; compose config passes."
}

if [[ "${1:-}" == "--help" && $# == 1 ]]; then
    usage
    exit 0
fi

if (( $# != 0 )); then
    usage >&2
    exit 2
fi

status=0

echo "Checking required files..."
for file in .dockerignore Dockerfile compose.yaml; do
    if [[ -f "$file" ]]; then
        echo "OK: $file"
    else
        echo "MISSING: $file" >&2
        status=1
    fi
done

if [[ -f compose.yaml ]]; then
    echo ""
    echo "Validating compose.yaml..."
    if docker compose config --quiet; then
        echo "OK: compose config valid"
    else
        echo "FAIL: compose config invalid" >&2
        status=1
    fi
fi

exit "$status"
