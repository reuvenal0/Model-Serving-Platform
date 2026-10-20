# Model Serving Platform

A Python portfolio project exploring object-oriented design, SOLID principles,
and a layered modular monolith for machine learning model serving.

## Current scope: Phase 4 - Deployment Lifecycle & Health Management

Phase 2 provides model and version registration, persistent SQLite metadata,
local binary artifact storage, artifact downloads, and version deletion through a
Python service. Phase 2.5 adds controlled logical deployment transitions and safe
routing configuration updates. Phase 3 adds real trusted sklearn/ONNX loading,
in-process prediction, and lazy, thread-safe runtime caching. Phase 4 adds persistent
logical deployments, guarded lifecycle orchestration, conservative readiness,
startup reconciliation, and deployment-aware deletion safety. Domain invariants
and the artifact subsystem are unchanged. No Phase 5 routing or APIs exist.

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

The suite retains all 421 tests from Phases 1 through 3. Execution tests build
small deterministic models locally, with no artifact downloads. Integration tests
use temporary SQLite databases and storage roots; event/barrier-controlled tests
verify cache concurrency and invalidation races without timing sleeps.

Run the repeatable demonstration after installing the project:

```bash
python examples/registry_demo.py
python examples/inference_demo.py
python examples/deployment_demo.py
```

The registry demo registers and retrieves harmless binary data without executing
it. The inference demo trains a trusted local sklearn model for `y = 2*x + 1`,
registers its Pickle bytes, predicts twice, verifies runtime identity reuse, and
prints cache statistics. It also builds and executes an equivalent ONNX graph
using the `onnx` development dependency. The deployment demo observes persisted
DEPLOYING, completes a model-specific readiness check, shows RUNNING, scales logical
replicas from 1 to 3, reads persisted metadata, stops, and rejects an illegal scale.
All demos dispose their engines and remove temporary data. None starts a server.

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

## Phase 3 execution architecture

```text
ModelRuntimeManager
  +-- ModelRepository -> version metadata
  +-- ArtifactStorage -> bytes at the stored logical key
  +-- ModelLoaderRegistry -> injected ModelFramework-to-ModelLoader mapping
        +-- SklearnModelLoader -> SklearnLoadedModel
        +-- OnnxModelLoader    -> OnnxLoadedModel
```

`LoadedModel` is a minimal protocol with `predict(inputs) -> outputs` and no
knowledge of loading, storage, or databases. `ModelLoader.load(artifact: bytes)`
creates one reusable runtime. The manager selects a loader through the injected
registry; it imports no sklearn, NumPy, ONNX Runtime, or concrete adapter. There
is no additional executor/factory hierarchy.

Explicit composition (see the inference demo for the repository and storage):

```python
loaders = ModelLoaderRegistry({
    ModelFramework.SKLEARN: SklearnModelLoader(trusted=True),
    ModelFramework.ONNX: OnnxModelLoader(),
})
manager = ModelRuntimeManager(
    repository, storage, loaders,
    max_cache_size=settings.runtime_cache_max_size,
)
runtime = manager.get_runtime("model-name", "v1")
predictions = runtime.predict([[1.0]])
```

### Lazy loading and cache lifecycle

Constructing the manager loads nothing. The first `get_runtime(model_name,
version)` retrieves metadata, requires status REGISTERED, retrieves bytes, selects
a loader, and loads a runtime. INVALID versions are rejected without changing
their status or touching artifacts. Cache keys are `(model_name, version)`;
registered artifacts and their metadata are assumed immutable while cached.
Hits reuse the same runtime and promote its LRU position. A completed load evicts
the least recently used entry when the positive integer capacity is exceeded.

The global bookkeeping lock is never held while reading metadata/storage,
deserializing models, waiting on other loads, or predicting. Requests for the same
uncached key share one loading future, including its error. Different keys can
load concurrently. Failed loads never enter the cache; the next request retries.
Loaders must support concurrent `load` calls for different keys. Built-in adapters
keep per-load state local and serialize predictions per runtime, using a separate
lock that does not block other models. Callers must not modify input buffers while
prediction is running or access/mutate private backend state.

`invalidate(name, version)` removes a completed entry and detaches any in-progress
load. Existing callers waiting for that load still receive its runtime or error,
but completion cannot restore the invalidated entry. A later request may start a
new generation immediately. Old success/failure cannot overwrite or remove that
new generation. `clear()` applies the same semantics to every tracked key. Neither
operation cancels native loading, closes runtimes, or invalidates a runtime held
by another caller. Eviction releases only cache references; Python/native memory
is not guaranteed to be reclaimed immediately.

`cache_info()` returns an immutable snapshot of lifetime `hits`, `misses`, `waits`,
`loads`, `load_failures`, and `evictions`, plus `current_size`, `max_size`, and
`in_flight`. Misses include waiting requests; loads count attempts by load owners.
`in_flight` counts current tracked generations, excluding detached work that can
still be running. `clear()` leaves the cumulative statistics intact.

### sklearn input contract and artifact trust

`SklearnModelLoader` accepts standard Pickle bytes containing a fitted sklearn
`BaseEstimator` with a callable `predict`. No paths or joblib-specific formats
are required/supported. Predictions delegate validation to the estimator and
return its native output (typically a NumPy array). For the demo's numeric
estimator, inputs are 2D feature lists or NumPy arrays of shape
`(samples, features)`; other supported shapes/types depend on the trained estimator
or pipeline. One loaded estimator is reused for repeated predictions.

**Only enable `trusted=True` for artifacts whose source and contents you trust.**
Pickle/joblib deserialization can execute arbitrary code. The flag records caller
acknowledgement; it does not authenticate data, inspect malicious payloads, or
provide a security sandbox. Loading is refused before deserialization without
that opt-in. Store trusted models in a private artifact root and use matching
training/serving dependencies. sklearn's incompatibility warnings are not
suppressed; cross-version model compatibility is not guaranteed. See sklearn's
[model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

### ONNX input/output contract

`OnnxModelLoader` creates one reusable CPU `InferenceSession` directly from bytes,
with one intra-op/inter-op thread per session. It discovers read-only input/output
metadata (names, ONNX tensor types, shape tuples) and supports self-contained
graphs with known tensor ranks and these dense input/output types:

| ONNX type | NumPy dtype |
| --- | --- |
| `tensor(float)` | `float32` |
| `tensor(double)` | `float64` |
| `tensor(int32)` | `int32` |
| `tensor(int64)` | `int64` |
| `tensor(bool)` | `bool` |

Pass a mapping of every input name to its tensor. For single-input graphs, a raw
NumPy array or nested Python list/tuple is also accepted. Arrays must already
have exactly the expected dtype. Numeric Python sequences are converted to the
declared dtype; strings are not coerced, boolean-only sequences cannot become
numeric features, and floating values cannot be truncated into integers. Integer range and
floating-conversion overflow are checked. Ranks, fixed dimensions, and consistent
shared symbolic dimensions are validated before session execution. Missing and
extra inputs are errors. Outputs are always `{output_name: numpy_array}`.

Strings, maps, sequences, sparse/optional tensors, unknown ranks, custom operators,
external tensor files, and GPU execution are outside the adapter contract. ORT
operator/opset support still applies. Graphs are executed in-process; this also
is not a sandbox. See the [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).

### Execution failures

Existing `ModelVersionNotFoundError`, `ArtifactNotFoundError`, and repository/storage
errors propagate unchanged. `UnsupportedFrameworkError` reports an unconfigured
or invalid framework. `ModelLoadingError` reports unusable/corrupted artifacts,
INVALID version status, or an invalid loader result; its subclass
`UntrustedArtifactError` reports missing Pickle trust. Backend prediction errors
and unsupported inputs raise `PredictionError`. Backend causes are chained rather
than discarded. Failures never silently return `None` or poison the runtime cache.

## Phase 4 deployment architecture

```text
DeploymentManager
  +-- DeploymentRepository -> SQLiteDeploymentRepository
  +-- ModelRepository     -> existing version identity/eligibility checks
  +-- ModelRuntimeManager -> existing shared lazy loading and LRU
  +-- DeploymentHealthChecker -> LoadedModel.predict (optional smoke check)
```

New implementation files are `app/interfaces/deployment_repository.py`,
`app/infrastructure/database/sqlite_deployment_repository.py`,
`app/services/deployment_manager.py`, and
`app/services/deployment_health_checker.py`. `DeploymentRecord` joins the existing
SQLAlchemy metadata. No model loader, runtime cache, or Domain lifecycle code was
duplicated. No new packages or configuration settings are required.

Compose a `SQLiteDeploymentRepository(sessionmaker(engine))` with the existing
SQLite engine helper and inject it, the existing model repository/runtime manager,
and a `DeploymentHealthChecker` into `DeploymentManager`. Both metadata repositories
must use the same database for the version-reference constraints to protect them.

### Persistent metadata and concurrency

The `deployments` table stores ID, model/version identity, replica count,
CPU/memory, status, optional endpoint metadata, aware UTC timestamps, and an integer
revision. Deployment IDs are unique and distinct from model/version identity;
multiple records may reference the same version. A composite foreign key points
to the registered `(model_name, version)` pair. The existing engine helper enables
foreign keys on every SQLite connection. Resources and timestamps reconstruct
existing Domain objects, including microseconds and UTC offsets.

The deployment repository returns `DeploymentSnapshot(deployment, revision)`;
revision tokens live outside the Domain. Initial saves are insert-only at revision
0. `update_deployment(deployment, expected_revision=...)` uses one conditional SQL
UPDATE to match the ID, revision, and immutable identity/creation fields and
increment the revision. A stale caller cannot overwrite a newer state, even
through a separate repository/engine/process using the same database. The manager
returns ordinary Domain entities and keeps tokens within each lifecycle operation.

Missing IDs raise `DeploymentNotFoundError`, duplicate saves raise
`DuplicateDeploymentError`, and stale updates raise `DeploymentConflictError`.
SQLite BUSY/LOCKED conflicts also raise `DeploymentConflictError`: re-read and
reconsider the operation before retrying. There are no blind retries. Other
backend errors raise `DeploymentPersistenceError` with their cause. Transactions
are scoped to repository calls; none spans model loading, waiting, or prediction.
No global management lock is used. The repository validates entity construction
but does not implement a separate transition state machine.

SQLite still has limited writer concurrency. These constraints/CAS updates protect
metadata; they are not distributed worker coordination or an execution lease.
Direct SQL writers that bypass revisions do not receive stale-write protection.
The setup follows SQLAlchemy's [composite foreign-key guidance](https://docs.sqlalchemy.org/en/20/core/constraints.html)
and [SQLite concurrency documentation](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html).

`initialize_database()` creates missing tables, including the new deployment table,
without altering existing ones. It is not a migration system. Existing databases
with incompatible layouts require an explicit migration before use; schema
versions, data backfills, and ALTER TABLE migration automation are not implemented.

### Lifecycle orchestration and failures

`deploy(model_name, version, resources, *, replicas=1, deployment_id=None,
smoke_inputs=...)` validates the request, verifies the model and a REGISTERED
version, allocates a UUID deployment ID unless supplied, and commits DEPLOYING.
It then obtains the runtime, checks readiness, invokes the Domain's `activate()`,
and commits RUNNING using the original revision. No endpoint URL is invented.

Loading/readiness failure invokes the Domain's `fail()` on a copy and attempts to
commit FAILED. The original error is re-raised after successful failure recording.
If committing RUNNING fails for a backend reason, FAILED is attempted against the
original DEPLOYING revision. A stale activation conflict is propagated without
retrying or overwriting another operation's state, including a scale that leaves
the record DEPLOYING. Inspect the record and reconcile or create a new deployment.

When failure-state persistence also fails, `DeploymentRecoveryError` contains
`deployment_id`, `operation_error`, and `persistence_error`, with the persistence
failure chained. The database can still contain DEPLOYING or an unverified RUNNING
record. Do not interpret it as healthy or silently ignore this error: restore
storage access and revalidate/reconcile. Uncertain commit outcomes likewise require
inspection; this is not crash-safe coordination between database and runtime work.

The exact Domain rules remain the table in **Controlled domain operations** below.
In particular, activation is DEPLOYING -> RUNNING; failure is DEPLOYING/RUNNING ->
FAILED; stop is DEPLOYING/RUNNING/FAILED -> STOPPED. Repeated stop is rejected with
`InvalidDeploymentStateError`; no automatic restart of FAILED/STOPPED records exists.
`scale(id, replicas)` invokes `Deployment.scale`, then persists with its read
revision. Failed validation leaves the saved snapshot untouched. Replicas remain
nonnegative integers, including zero, and are **logical metadata only**: changing
1 to 3 launches no processes, containers, sessions, or extra execution capacity.
Resource requests are also metadata, not enforced CPU/memory quotas.

`undeploy(id)` invokes `stop()` and commits STOPPED, preserving history, resources,
and replica count. It does not evict/close a runtime or terminate active calls.
`get_deployment(id)` and `list_deployments(model_name=None)` return Domain snapshots;
listing sorts by ID and a filter with no matches returns an empty list.

### Readiness and shared cache behavior

Persisted RUNNING status is lifecycle intent, not process readiness. Use
`check_health(id)` before treating a deployment as ready. For DEPLOYING, FAILED,
or STOPPED, it returns `ready=False` without loading/predicting. For RUNNING it
obtains the current shared runtime (lazily reloading after eviction), checks it,
and re-reads the revision before returning `DeploymentHealth`. If a concurrent
operation changed the record, it raises a conflict instead of reporting stale
readiness. This is a point-in-time observation, not an inference authorization
lease; callers must coordinate any later use. There is no deployment inference API.

The default `load` check proves that a runtime was obtained and exposes predict;
it does **not** call predict or synthesize a universal feature matrix. Passing
explicit `smoke_inputs` to deploy or check_health invokes one genuine prediction
and reports check kind `smoke`. Inputs must be valid, trusted, model-specific
fixtures (including the existing ONNX named-input contract). Successful completion
does not prove accuracy or that every future input will work. Smoke inputs are
not persisted; restart checks default to load-readiness. The checker itself never
loads/deserializes artifacts and inherits the trusted-source restrictions of the
injected loaders.

Load/smoke failure during RUNNING health checks attempts to persist FAILED with
optimistic concurrency; original errors and failed recovery remain explicit. A
cached runtime alone never establishes deployment status. Deployments sharing a
version share cache identity; stopping one leaves others operational. Eviction
changes no deployment metadata, and no pinning or reference-counting is added.
Logical replica count (including zero) is not runtime-residency or capacity proof.

### Startup reconciliation

Call `manager.reconcile()` explicitly during **quiescent single-process startup,
before accepting work**. It revalidates persisted RUNNING records against the
current runtime manager, marks interrupted DEPLOYING records FAILED, and leaves
FAILED/STOPPED records inactive. It never automatically restarts them. A fresh
process has an empty cache and must reload its RUNNING versions; no persisted
readiness evidence is trusted.

The returned `ReconciliationReport` has `health` observations and explicit
`failures` containing deployment IDs and original/recovery errors. Processing
continues after individual failures; callers must inspect failures before
advertising readiness. A list/read failure that prevents reconciliation itself
propagates. There is no background monitor, process lease, multi-process startup
arbitration, or automatic recovery retry. Do not run this startup sweep against
another process's live deployment attempts. Regular lifecycle updates still use
revisions to prevent stale completions from restoring RUNNING.

### Deployment-aware version deletion

The deletion policy is deliberately conservative: **every deployment history
reference blocks version deletion**, including DEPLOYING, RUNNING, FAILED, and
STOPPED records. Active states must retain their artifacts; inactive history must
remain reconstructable through its foreign key. Archiving history and relaxing
this restriction require a future explicit retention policy.

SQLite enforces `ON DELETE RESTRICT` on the composite foreign key. This is atomic
with metadata writes and protects against a deployment appearing after a registry
precheck: if deployment insertion wins, deletion is rejected; if deletion wins,
a later deployment insert fails its parent-version foreign key. Lock contention
may instead require retrying after a fresh read. Integration tests cover both
orderings and independent connections. No check-then-delete service coupling is
used. `SQLiteModelRepository.delete_version` narrowly translates a rejected delete
into `ModelVersionInUseError`. The existing ModelRegistry already deletes metadata
before bytes, so it propagates that error while preserving the artifact.
Versions with no deployment history retain the original deletion behavior.

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
- Deployment history prevents version deletion even after stopping. No history
  archival/purge policy is implemented.
- Model loading and prediction run inside the application process, with no worker
  subprocess, process isolation, timeout, cancellation, or resource enforcement.
  A bad artifact/backend can exhaust resources or crash the process.
- Runtime caches are process-local and count models, not memory bytes. Concurrent
  in-progress loads are not capacity-limited. Each runtime serializes its own
  predictions, favoring safety over throughput.
- Registry deletion/re-registration does not automatically notify runtime caches.
  Callers must invalidate affected entries or clear caches and coordinate any
  concurrent registry changes. Existing users of a detached runtime may finish.
- Deployment currently means logical readiness in this application instance.
  Remote/container orchestration, real replica workers, REST APIs, traffic routing,
  metrics, authentication, background workers, cloud adapters, and distributed
  coordination are not implemented. The `api` package remains empty.

## Domain conventions

- Dataclasses validate at construction and raise `DomainValidationError` for
  invalid domain values. `Model`, `ModelVersion`, and `InferenceLog` remain mutable
  snapshots. The repository revalidates models and versions before writing them.
  Deployment and routing fields reject direct assignment; their explicit methods
  control changes while preserving invariants.
- `ResourceConfig` remains an immutable value object. Routing weights are copied
  on construction and replacement and exposed as a read-only mapping.
- Required identifiers reject blank strings without changing supplied names.
  Version identifiers and artifact locations remain opaque strings.
- All timestamp defaults are aware UTC datetimes. Supplied aware timestamps are
  normalized to UTC; naive timestamps are rejected. Updates to `updated_at` are
  explicit for registry snapshots. Successful deployment operations and routing
  updates refresh `updated_at` automatically.
- Enum members are required at construction; raw string conversion belongs at
  future system boundaries. Enum values use lowercase strings.
- Numeric counts and weights reject booleans and fractional values; CPU and
  latency must be finite. Zero replicas, zero latency, and zero individual
  routing weights are allowed where the other invariants hold.
- Schema dictionaries have independent defaults. Schema contents are opaque
  metadata; no schema language or schema validation is imposed yet.
- Inference logs store request metadata and an optional input hash, never raw
  input payloads.

## Controlled domain operations

`Deployment.activate()`, `fail()`, and `stop()` record logical state changes.
They perform no external deployment, health checks, cleanup, or orchestration.
DeploymentManager obtains the shared runtime, checks readiness, and records the
logical result through these methods; it performs no remote orchestration. Allowed transitions are deliberately limited:

| Current status | `activate()` | `fail()` | `stop()` | `scale(replicas)` |
| --- | --- | --- | --- | --- |
| DEPLOYING | RUNNING | FAILED | STOPPED | Allowed |
| RUNNING | Rejected | FAILED | STOPPED | Allowed |
| FAILED | Rejected | Rejected | STOPPED | Rejected |
| STOPPED | Rejected | Rejected | Rejected | Rejected |

Repeated transitions are rejected with `InvalidDeploymentStateError`, including
repeated stopping. Retry/restart policy is deferred; no method implicitly restarts
a failed or stopped deployment. Construction still accepts all existing enum
statuses to support restoring snapshots. Replica counts remain nonnegative
integers, including zero; booleans and fractional counts are rejected with
`DomainValidationError` before state eligibility is checked. Scaling changes only
the logical count. State changes preserve identity, resources, endpoint, replicas,
and `created_at`; scaling preserves status. Successful operations refresh
`updated_at` in UTC. Rejected operations leave the entity unchanged.

`RoutingConfig.update_weights(weights)` validates and copies a complete
replacement before publishing it and refreshing `updated_at`. The same rules
apply to construction and updates: a nonempty mapping, nonblank version names,
integer weights from 0 to 100, and a total of exactly 100. Invalid updates preserve
both existing weights and timestamp. Input dictionaries and previously obtained
read-only weight views remain independent snapshots. No version lookup or routing
algorithm is performed here.

These two classes use frozen dataclasses to reject public assignment and deletion.
Only their domain methods use internal assignment to update validated state. They
are explicitly unhashable because these methods change their values. This is
protection against accidental mutation, not a security boundary or thread lock.
Whole-object reconstruction with `dataclasses.replace` remains validated and is
available for restoring snapshots or supplying a deployment endpoint.

Compatibility: constructor argument names/order, default statuses, and attribute
reads are preserved. Direct field assignment is intentionally no longer supported
for deployments and routing configurations. `weights` is now a read-only `Mapping`
rather than a mutable dictionary; use `dict(config.weights)` when a mutable or
JSON-serializable export is needed. Generic `dataclasses.asdict`, deepcopy, and
pickle handling of mapping proxies is not supported; serializers must export
weights explicitly. The existing registry uses neither deployment nor routing
serialization, so no ORM or registry changes are needed.

`ModelVersion` still describes registration status (REGISTERED or INVALID), not a
deployment lifecycle. No status methods were added because validation execution
and a lifecycle policy have not been specified. `Model`, `ResourceConfig`, and enums remain unchanged. Application exceptions
cover deployment persistence and recovery.

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
| `RUNTIME_CACHE_MAX_SIZE` | `16` (positive integer; completed runtime entries) |

The database URL and artifact directory configure the adapters when explicitly
constructed. Importing configuration initializes no services. The upload-size
setting is reserved for a future upload boundary and is not enforced. Pass
`settings.runtime_cache_max_size` to the runtime manager's `max_cache_size`
constructor parameter; the manager does not read environment variables itself.
