# Model Serving Platform

A Python portfolio project exploring object-oriented design, SOLID principles,
and a layered modular monolith for machine learning model serving.

## Current scope: Phase 1 — Foundation & Domain

The domain contains model identities, version registration metadata, logical
deployments, immutable resource requirements, traffic weight configurations,
and inference request metadata. String enums describe supported frameworks and
independent version and deployment states. Domain errors share a small exception
hierarchy in `app/core/exceptions.py`.

The domain uses only the Python standard library and application domain errors.
It has no FastAPI, Pydantic, SQLAlchemy, or infrastructure dependencies. The
`interfaces`, `services`, `infrastructure`, and `api` packages reserve the layer
boundaries and intentionally contain no implementations.

There are no endpoints, repositories, persistence, artifact storage operations,
model execution, routing algorithms, or deployment orchestration in this phase.
Cross-entity uniqueness and repository-dependent rules belong to Phase 2.

## Setup and tests

Requires Python 3.11 or newer. A virtual environment is recommended:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Domain conventions

- Dataclasses validate at construction and raise `DomainValidationError` for
  invalid domain values. Entities are mutable snapshots; future application
  operations must construct validated replacements when changing validated fields.
- `ResourceConfig` is a frozen value object. Routing weights are copied on
  construction to prevent caller dictionary changes from affecting the snapshot.
- Required identifiers reject blank strings without changing supplied names.
  Version identifiers and artifact locations remain opaque strings.
- All timestamp defaults are aware UTC datetimes. Supplied aware timestamps are
  normalized to UTC; naive timestamps are rejected. Updates to `updated_at` are
  explicit, with no automatic lifecycle behavior in this phase.
- Enum members are required at construction; raw string conversion belongs at
  future system boundaries. Enum values use lowercase strings.
- Numeric counts and weights reject booleans and fractional values; CPU and
  latency must be finite. Zero replicas, zero latency, and zero individual
  routing weights are allowed where the other invariants hold.
- Schema dictionaries have independent defaults. Schema contents are opaque
  metadata; no schema language or schema validation is imposed yet.
- Inference logs store request metadata and an optional input hash, never raw
  input payloads.

## Configuration

Construct `app.config.Settings()` to read environment values. Importing the
module creates no directories, database connections, or external services.
Explicit constructor values override environment defaults.

| Environment variable | Default |
| --- | --- |
| `APP_NAME` | `Model Serving Platform` |
| `ARTIFACT_STORAGE_DIR` | `artifacts` (relative to the working directory) |
| `DATABASE_URL` | `sqlite:///./model_serving.db` |
| `MAX_UPLOAD_SIZE_BYTES` | `104857600` (100 MiB; positive integer) |

These are configuration values only; upload limits and storage operations are
not executed in Phase 1.
