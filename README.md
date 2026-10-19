# Model Serving Platform

A Python portfolio project exploring object-oriented design, SOLID principles,
and a layered modular monolith for machine learning model serving.

## Current scope: Phase 2 — Model Registry & Persistence

Phase 1 domain entities and validation rules remain unchanged. Phase 2 adds model
and version registration, persistent SQLite metadata, local binary artifact
storage, artifact downloads, and version deletion through a Python service.

```text
ModelRegistry
  +-- ModelRepository (ABC) -> SQLiteModelRepository -> SQLite
  +-- ArtifactStorage (ABC) -> LocalArtifactStorage -> local files
```

The Repository Pattern separates metadata operations from database details.
`ModelRegistry` receives both interfaces through constructor injection and has no
SQLAlchemy or filesystem dependencies. Concrete adapters are assembled explicitly
by the caller. The domain uses only the Python standard library and core domain
errors. No global sessions, dependency injection container, or server are needed.

Key implementation files:

- `app/interfaces/model_repository.py`: insert-only metadata contract.
- `app/interfaces/artifact_storage.py`: logical keys and bytes storage contract.
- `app/infrastructure/database/database.py`: SQLite engine and schema initialization.
- `app/infrastructure/database/records.py`: internal SQLAlchemy mappings and UTC type.
- `app/infrastructure/database/sqlite_model_repository.py`: domain/persistence mapping.
- `app/infrastructure/storage/local_artifact_storage.py`: safe local artifact adapter.
- `app/services/model_registry.py`: registration, queries, downloads, and deletion.
- `app/core/exceptions.py`: domain errors and explicit backend/consistency failures.

## Setup and tests

Requires Python 3.11 or newer. A virtual environment is recommended:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[dev]"
python -m pytest -q
```

The suite includes the original 96 Phase 1 tests, registry unit tests with mocked
interfaces, and integration tests using temporary SQLite databases and artifact
roots. It tests foreign keys, independent repository instances, UTC precision,
path safety, complete binary content, collisions, and compensating-action failures.

Run the repeatable demonstration after installing the project:

```bash
python examples/registry_demo.py
```

It initializes temporary storage, injects both adapters, registers a sample model
and version, lists them, downloads harmless binary data, and verifies exact byte
identity. Its database engine is disposed and temporary data is removed on exit.
It never loads or executes the artifact.

## Metadata persistence

Use `create_sqlite_engine(settings.database_url)`, explicitly call
`initialize_database(engine)`, then construct
`SQLiteModelRepository(sessionmaker(engine))`. The engine helper must be used for
SQLite connection and transaction configuration. Dispose the engine when done.
See the demo for a complete runnable composition using the existing `Settings`.

Each repository operation opens and closes its own transaction through the
injected SQLAlchemy session factory. Saves insert records and commit before
returning; they never update existing records. Failures roll back. Unique database
constraints protect model names and `(model_name, version)` pairs. Every connection
enables SQLite foreign-key enforcement; versions reference existing models, and
parents cannot be deleted while versions reference them.

Repository get operations raise `ModelNotFoundError` or
`ModelVersionNotFoundError`. Saves raise the corresponding duplicate error;
`save_version` also requires an existing parent. Lists sort lexically by name or
version. Listing an existing model with no versions returns `[]`; listing versions
of a missing model raises `ModelNotFoundError`. Deleting a missing version raises
`ModelVersionNotFoundError`. A successful repository deletion returns the domain
snapshot removed within that transaction, which supplies the precise artifact key
for service cleanup. Other database failures become `RepositoryError`, with the
backend exception preserved as the cause. ORM records never leave the adapter.

Timestamps are stored as ISO strings with an explicit UTC offset and restored as
aware UTC datetimes, retaining microseconds. No naive domain timestamps are
accepted. Mutable domain snapshots are revalidated before saves. Schema metadata
must be JSON-compatible; the registry normalizes JSON containers (for example,
tuples become arrays) before any storage writes.

The transaction and connection setup follows SQLAlchemy's
[session transaction guidance](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
and [SQLite documentation](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html).

## Local artifact storage

`ArtifactStorage` exposes `save(key, bytes)`, `retrieve(key) -> bytes`,
`delete(key)`, and `exists(key) -> bool`. Callers use logical keys, never local
paths. Saves are create-only; duplicates raise `DuplicateArtifactError` without
replacing bytes. Missing retrievals/deletions raise `ArtifactNotFoundError`.
Invalid keys raise `DomainValidationError`; backend failures raise
`ArtifactStorageError`.

`LocalArtifactStorage(root)` creates its configured root and accepts relative
slash-separated keys whose components contain ASCII letters, digits, `_`, `-`,
and internal `.` characters. Empty components, traversal, absolute paths,
backslashes, drive names, null bytes, and unsafe characters are rejected. Each
key maps to a flat SHA-256 filename under the canonical root, so key components
never become physical directories. Target symlinks/reparse points are rejected.

Bytes are written to a temporary file in that root, flushed, and published
without replacement: Windows uses its non-overwriting rename; POSIX uses a hard
link and removes the temporary name. A failed write/publication cleans up its
partial temporary file. An exceptional temporary cleanup failure raises
`ArtifactCleanupError`, including the temporary path and whether publication
succeeded. The storage root must be private and trusted; portable checks do not
protect against a hostile local process replacing paths between checks.

## Registration and consistency

`register_model(name, description="")` returns the registered `Model`.
`register_version(model_name, version, framework, content, *, input_schema=None,
output_schema=None)` returns the registered `ModelVersion` with status REGISTERED.
`framework` must be a `ModelFramework` member and `content` must be bytes.
`artifact_location` stores the logical key, not a filesystem path.

Registration allocates an opaque UUID key, validates metadata and content, checks
the parent and duplicate version, saves bytes, then commits version metadata.
The key is collision-resistant and remains stable in the persisted metadata.
If metadata fails, the registry deletes only the artifact whose save succeeded in
that operation. An artifact collision does not authorize cleanup of the existing
key. Database uniqueness constraints also reject a version inserted after the
service's initial check.

A successful compensation re-raises the original registration error. A failed
compensation raises `RegistryConsistencyError`, exposing `artifact_key`,
`operation_error`, and `cleanup_error`. Temporary artifact cleanup failures also
become explicit consistency errors before metadata is saved. Inspect the original
`ArtifactCleanupError` for its temporary path and publication flag.

`delete_version(model_name, version)` verifies existence, commits metadata deletion,
then deletes the artifact identified by the repository's deleted snapshot. It
returns `None` and preserves the parent model. If metadata deletion fails, storage
is left alone. If artifact deletion fails, metadata remains deleted and
`RegistryConsistencyError` reports the key and cleanup error. A preexisting missing
artifact is also reported explicitly. Do not restore metadata that may reference
unavailable bytes; inspect storage and retry `storage.delete(error.artifact_key)`
directly if the artifact remains. Retrying registry deletion after this partial
failure will report a missing version.

## Current limitations

- SQLite and filesystem operations are separate transactions. Compensating actions
  handle ordinary exceptions; they are not crash-safe distributed transactions.
  Crashes, uncertain commit outcomes, or failed cleanup may require manual metadata
  and artifact reconciliation. Temporary leftovers are never silently hidden.
- Individual repository operations and artifact publication are atomic, but the
  full registry workflow is not isolated across backends. Concurrent downloads
  can observe deletion, and complex concurrent workflows need caller coordination.
- Schema initialization creates missing tables; migrations are not implemented.
- Artifacts use an in-memory bytes API. Streaming and configured upload-size
  enforcement are not implemented. POSIX storage requires hard-link support.
- Deployment-aware deletion protection belongs to Phase 4; no deployment checks
  exist yet.
- Model loading/execution, inference, deployment orchestration, REST APIs, traffic
  routing, metrics, authentication, background workers, and cloud adapters are not
  implemented. The `api` package remains empty.

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

The database URL and artifact directory configure the adapters when explicitly
constructed. Importing configuration initializes no services. The upload-size
setting is reserved for a future upload boundary and is not enforced in Phase 2.
