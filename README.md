# Model Serving Platform

A Python portfolio project exploring object-oriented design, SOLID principles,
and a layered modular monolith for machine learning model serving.

## Current scope: Phase 7.5 - Runtime Recovery & Resource Safety

Phase 2 provides model and version registration, persistent SQLite metadata,
local binary artifact storage, artifact downloads, and version deletion through a
Python service. Phase 2.5 adds controlled logical deployment transitions and safe
routing configuration updates. Phase 3 adds real trusted sklearn/ONNX loading,
in-process prediction, and lazy, thread-safe runtime caching. Phase 4 adds persistent
logical deployments, guarded lifecycle orchestration, conservative readiness,
startup reconciliation, and deployment-aware deletion safety. Domain invariants
and the artifact subsystem are unchanged. Phase 5 adds explicit-version inference
and a small FastAPI transport. Phase 6 adds deployment-aware weighted routing,
canary allocations, revisioned routing persistence, and validated rollback. Phase 7
adds persistent sanitized inference observations and windowed metrics through
local monitoring endpoints. Phase 7.5 adds automatic startup reconciliation, bounded
runtime artifact retrieval, and process/cache resource visibility.

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

The suite retains all 507 tests from Phases 1 through 4. Execution tests build
small deterministic models locally, with no artifact downloads. Integration tests
use temporary SQLite databases and storage roots; event/barrier-controlled tests
verify cache concurrency and invalidation races without timing sleeps.

Run the repeatable demonstration after installing the project:

```bash
python examples/registry_demo.py
python examples/inference_demo.py
python examples/deployment_demo.py
python examples/api_demo.py
```

The registry demo registers and retrieves harmless binary data without executing
it. The inference demo trains a trusted local sklearn model for `y = 2*x + 1`,
registers its Pickle bytes, predicts twice, verifies runtime identity reuse, and
prints cache statistics. It also builds and executes an equivalent ONNX graph
using the `onnx` development dependency. The deployment demo observes persisted
DEPLOYING, completes a model-specific readiness check, shows RUNNING, scales logical
replicas from 1 to 3, reads persisted metadata, stops, and rejects an illegal scale.
The API demo creates an application lifespan and uses TestClient to register,
deploy, predict, batch predict, stop, and verify post-stop rejection. It verifies
owned cache cleanup on shutdown. All demos dispose their engines and remove
temporary data. None requires an external server.

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
without altering existing ones. Phase 6 additionally runs the numbered routing
migration described below. Incompatible legacy layouts still require explicit
operator migration; no general ALTER TABLE/backfill automation is provided.

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
lease; callers must coordinate any later use. Phase 5's InferenceService performs
its own final revision admission check before executing a request.

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

FastAPI lifespan now calls reconciliation automatically during **quiescent
single-process startup, before accepting work** (Phase 7.5). Standalone Python
applications can still call `manager.reconcile()` explicitly. It revalidates persisted RUNNING records against the
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

## Phase 5 inference and HTTP architecture

```text
HTTP -> FastAPI synchronous router -> InferenceService
                                      +-- ModelRepository
                                      +-- DeploymentRepository (revision snapshots)
                                      +-- DeploymentHealthChecker
                                      +-- ModelRuntimeManager -> LoadedModel.predict
```

The router handles transport/Pydantic DTOs, not SQL or filesystem operations.
`InferenceService` performs model/version checks, bounded tensor validation,
deployment admission, runtime readiness, prediction, and output normalization.
`PredictionData` is its data-format utility; NumPy conversion remains outside the
Domain. The service never invokes loaders or receives ORM objects. Existing
Domain, registry, persistence, and runtime-manager semantics remain intact.
The adapters add a `PredictionInputError` subtype of the existing `PredictionError`
to distinguish input errors from execution failures without breaking existing
callers that catch the base class.

New files include `app/services/inference_service.py`,
`app/services/inference_data.py`, `app/core/container.py`, `app/main.py`,
`app/api/dependencies.py`, `app/api/routers/inference.py`,
`app/api/schemas/inference.py`, `app/api/errors.py`, and `app/api/middleware.py`.

### Running the API and ownership

After installing `.[dev]`, start locally:

```bash
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Swagger documentation is at `/docs`; `/openapi.json` includes the request,
response, and structured error schemas. Explicit versions preserve the Phase 5
behavior; Phase 6 also permits omitted versions using configured weighted routing.
Register/deploy models through Python services (see the API demo); registration,
upload, and deployment mutation REST endpoints remain outside the current scope.

Importing `app.main` and calling `create_app(settings)` allocate no application
graph or database/storage resources. Lifespan startup constructs an owned
`ServiceContainer` in a worker thread, with the engine, session factory,
repositories, local storage, loaders/registry, runtime manager, health checker,
deployment manager, model registry, and inference service. No global sessions
exist. On shutdown the owned runtime cache is cleared and the engine disposed.
An explicitly supplied `create_app(container=container)` uses caller-owned resources;
close that container explicitly or use its context manager. Dependencies can be
overridden with `app.dependency_overrides[get_inference_service]` or
`get_deployment_manager`. Tests use temporary **file-backed** SQLite databases;
the existing in-memory engine is not shared across the HTTP thread pool.

Phase 7.5 lifespan startup runs reconciliation before exposing its container to
requests. This requires a single quiescent lifecycle owner; do not start another
application process against live deployment attempts. Each inference request still
validates runtime readiness and deployment revision. Persisted RUNNING alone is
never used as verified readiness.

Lifespan ownership follows [FastAPI's lifecycle guidance](https://fastapi.tiangolo.com/advanced/events/).
Routes use ordinary `def`, so blocking loading/prediction runs through FastAPI's
[external worker thread pool](https://fastapi.tiangolo.com/async/). Pool workers are
shared across requests/dependencies and finite; requests may queue. Per-runtime
prediction locks and native ML thread pools may further constrain concurrency.
There are no distributed limits, prediction timeouts, cancellation guarantees,
or process isolation. Servers must drain requests before closing owned resources.

### Endpoints and wire format

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/predict/{model_name}` | One vectorized call using an explicit or routed version |
| POST | `/predict/{model_name}/batch` | Explicit alias for the same bounded batched-tensor call |
| GET | `/health` | API process liveness only: `{"status":"ok"}` |
| GET | `/deployments/{deployment_id}/health` | Existing load-readiness check and lifecycle state |

Example sklearn request (after registering/deploying a matching one-feature model):

```json
{"version": "v1", "inputs": [[3.0], [4.0]]}
```

Illustrative response (latency and floating-point rounding vary):

```json
{"model_name": "linear-demo", "version": "v1", "predictions": [7.0, 9.0], "latency_ms": 2.4}
```

The batch endpoint accepts exactly the same envelope. Both allow one or more
samples up to `MAX_BATCH_SIZE`; the explicit batch URL does not mean a collection
of independent prediction requests. One runtime retrieval and one backend predict
call handle the entire tensor. Backend sample ordering is preserved. Different
samples in a tensor must have the same shape and scalar category; integer and
floating JSON numbers may mix for floating models. Multiple named tensors must
share the same leading sample count.

sklearn HTTP inputs are nonempty rectangular **2D finite numeric lists** of shape
`[samples, features]`. Strings, nulls, booleans, ragged arrays, and scalars are
rejected; this API does not expose every custom estimator input format. Returned
numeric, boolean, or string label arrays are normalized without stringifying data.

ONNX accepts a nonempty rectangular numeric/boolean list for a single input, or a
mapping such as `{"X": [[1.0, 2.0]], "mask": [[true, false]]}`. Tensor ranks are
1 through the configured maximum; rank-zero tensor HTTP inputs are not exposed.
All graph input names are required. The existing loader enforces declared rank,
fixed/symbolic shapes, and float32/float64/int32/int64/bool types. JSON lists carry
no NumPy dtype tag: floating tensors use the loader's numeric conversion, while
integer tensors reject fractions/overflow and boolean tensors reject numeric
coercion. String tensors remain unsupported. Native callers passing NumPy arrays
directly to a loaded runtime still follow Phase 3's stricter ndarray dtype rules;
InferenceService itself accepts the documented JSON list/mapping contract.

Outputs support NumPy arrays/scalars and nested lists/tuples, string-keyed objects,
finite native numbers, strings, booleans, and null. ONNX output names are preserved
as object keys. Complex values, arbitrary objects, bytes, non-string keys, and
nonfinite floats raise a serialization error; oversized/nested outputs are
rejected within the configured element/rank bounds. Normalization does not make
arbitrary model execution memory-safe. `latency_ms` uses a monotonic clock and
includes service validation, metadata/readiness/admission, possible loading,
prediction, and normalization; it is not a pure backend timing metric.

### Optional tensor contracts

Existing schema metadata is opaque unless it contains an `inference` key. This
small explicit contract is **not arbitrary JSON Schema**:

```python
input_schema = {"inference": {"shape": [None, 1], "dtype": "number"}}
output_schema = {"inference": {"shape": [None], "dtype": "number"}}
# Named tensors, for example ONNX outputs:
output_schema = {"inference": {"tensors": {
    "Y": {"shape": [None, 2], "dtype": "number"}
}}}
```

Each tensor contract requires exactly `shape` and `dtype`. Shape is a list of
nonnegative fixed dimensions or `None` wildcards, up to the rank bound. Dtype is
`number`, `integer`, `boolean`, or `string`; `number` accepts integer JSON values.
These are JSON scalar categories, not floating bit-width declarations. A named
contract requires exactly `tensors` and an exact, nonempty name-to-contract map.
Unknown contract keywords and malformed specifications are model configuration
errors. Other metadata keys (including old `type`/`features` descriptions) remain
uninterpreted. Output contracts apply after normalization to JSON lists, so empty
lists cannot communicate unobserved trailing dimensions.

Input mismatches return 422; unsupported contract configuration returns 503;
output mismatches/serialization failures return 500. Actual estimator/graph
validation still applies. sklearn ValueError/TypeError prediction failures are
classified as input failures; the adapter cannot distinguish every custom
estimator bug that uses those exception types. Other backend failures remain
prediction failures with their original chained cause.

### Admission consistency and failures

The service verifies the model and REGISTERED version, validates inputs, and
requires a RUNNING deployment for that exact version. It obtains the runtime once
and runs the existing conservative readiness checker without a fabricated smoke
prediction. It then freshly reads a candidate deployment and admits the request
only if its identity, RUNNING status, and original revision still match. Another
stable RUNNING deployment of the same version may admit it if an earlier candidate
changed. This deterministic eligibility check does not route between versions.

**Admission is the successful final snapshot read.** A stop committed before that
read is rejected; a changed RUNNING revision (for example scaling) requires a new
admission attempt and returns 409 if no stable candidate exists. Requests already
admitted may finish after undeploy. No transaction or global lifecycle lock spans
prediction, and this is not strict cross-process atomic execution admission or a
lease. Concurrency tests cover stopping before admission, scaling during readiness,
and completing an admitted prediction while later requests are rejected.

Inference input/execution errors do not mutate deployment status. In particular,
a bad client's feature matrix does not mark a shared deployment FAILED. Loading
failure denies admission and returns 503; the deployment-health endpoint retains
Phase 4's failure-recording/recovery behavior. Existing model-version deletion and
manual runtime invalidation policies remain unchanged.

Errors use a stable public envelope without exception text, filesystem paths,
raw inputs, stack traces, or backend details:

```json
{"error": {"code": "model_unavailable", "message": "Model or supporting service is unavailable"}}
```

| Status | Meaning |
| --- | --- |
| 404 | Missing model, version, deployment, or route |
| 409 | Known lifecycle/revision conflict |
| 413 | Body exceeds the configured byte limit |
| 422 | Malformed request, input structure, shape, dtype, or input contract mismatch |
| 503 | No eligible deployment, unavailable runtime/backend, or invalid model contract |
| 500 | Unexpected execution, unsupported output, or internal failure |

Required identifiers in HTTP paths/version fields are nonblank and at most 200
characters. Pydantic rejects extra body fields and non-string versions. ASGI
middleware checks both declared Content-Length and actual received bytes before
JSON parsing, including requests without a length header. It buffers at most the
configured accepted body size (plus server-owned incoming chunks). Prediction
endpoints accept JSON only, with no artifact/Pickle upload capability. Health
liveness does not load models or claim every deployment is ready.

The container defaults to **TRUST_SKLEARN_ARTIFACTS=false**. Enable it only after
establishing trust in the artifact source/root; it is acknowledgement, not a
sandbox or provenance verifier. All execution remains in-process. The API has no
authentication, tenant isolation, production rate limiting, or worker isolation;
those are outside this phase. Phase 6 routing management endpoints are strictly
for local development until authorization and authentication are implemented.

## Current limitations

- SQLite and filesystem operations are separate transactions. Compensating actions
  handle ordinary exceptions; they are not crash-safe distributed transactions.
  Crashes, uncertain commit outcomes, or failed cleanup may require manual metadata
  and artifact reconciliation. Temporary leftovers are never silently hidden.
- Individual repository operations and artifact publication are atomic, but the
  full registry workflow is not isolated across backends. Concurrent downloads
  can observe deletion, and complex concurrent workflows need caller coordination.
- Legacy tables retain their initialization behavior; a numbered additive migration
  installs routing tables. General schema evolution/backfill tooling is not implemented.
- Artifacts use an in-memory bytes API. Streaming and configured upload-size
  enforcement are not implemented. POSIX storage requires hard-link support.
- Deployment history prevents version deletion even after stopping. No history
  archival/purge policy is implemented. Routing history also retains version references,
  including zero-weight versions, and blocks deletion without controlled history cleanup.
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
  Remote/container orchestration, real replica workers, authentication,
  background workers, cloud adapters, and distributed coordination are not implemented.
  REST endpoints cover prediction, health, local routing management, and monitoring.

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
  metadata. Only the explicit Phase 5 `inference` tensor-contract namespace is
  validated; this does not implement arbitrary JSON Schema.
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
| `MAX_RUNTIME_ARTIFACT_BYTES` | `104857600` (100 MiB; positive integer; runtime artifact retrieval) |
| `MAX_REQUEST_BODY_BYTES` | `1048576` (1 MiB; HTTP bytes before JSON parsing) |
| `MAX_BATCH_SIZE` | `128` (samples per tensor, both prediction endpoints) |
| `MAX_TENSOR_DIMENSIONS` | `4` (tensor rank) |
| `MAX_TENSOR_ELEMENTS` | `100000` (total scalar input/output budget) |
| `TRUST_SKLEARN_ARTIFACTS` | `false` (`true`/`false`, case-insensitive) |

The database URL and artifact directory configure the adapters when explicitly
constructed. Importing configuration initializes no services. The upload-size
setting is reserved for a future upload boundary and is not enforced. Pass
`settings.runtime_cache_max_size` to the runtime manager's `max_cache_size`
constructor parameter; the manager does not read environment variables itself.

## Persistent development API model

Run these PowerShell commands from the project root, after installing `.[dev]`.
Both processes must use the same working directory and environment settings:

```powershell
$env:TRUST_SKLEARN_ARTIFACTS = 'true'
python -m examples.seed_development_model
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

In another terminal, from the project root:

```powershell
python -m examples.test_live_api
```

The seed prints the resolved SQLite filename, artifact root, logical artifact key,
and deployment ID. Defaults are `./model_serving.db` and `./artifacts`, resolved
against the working directory. Existing `DATABASE_URL` and `ARTIFACT_STORAGE_DIR`
environment settings are honored unchanged by both seed and server. No `.env`
file is loaded automatically. Persistent data remains after either process exits.

Open [Swagger UI](http://127.0.0.1:8000/docs), select `POST /predict/{model_name}`,
enter `demo-linear`, and submit:

```json
{"version": "v1", "inputs": [[4.0]]}
```

The response resembles the following (latency and floating-point rounding vary):

```json
{"model_name": "demo-linear", "version": "v1", "predictions": [9.0], "latency_ms": 0.5}
```

For `POST /predict/demo-linear/batch`, submit
`{"version":"v1","inputs":[[4.0],[5.0]]}` and expect predictions approximately
`[9.0,11.0]`. Both endpoints use vectorized tensors; batch preserves row order.
The live script checks liveness, deployment readiness, prediction, and batch
prediction, printing HTTP statuses, JSON, and PASS/FAIL with a nonzero failure exit.

The seeder registers only missing records and checks existing metadata and exact
artifact bytes against a newly trained deterministic estimator **before loading**.
A mismatch is an error, never an overwrite. Dependency upgrades may change Pickle
bytes; inspect mismatches manually rather than bypassing this safeguard. Run one
seeder at a time without concurrent modification of demo records or storage.
It reuses a valid RUNNING demo deployment. Inactive deployments remain unchanged;
new ones use `demo-api-deployment-2`, `-3`, and so on. Subsequent seeds reuse an
already RUNNING replacement. For a replacement, use the printed command:
`python -m examples.test_live_api --deployment-id demo-api-deployment-2`.
ID collisions with different configuration require manual inspection.

Pickle can execute arbitrary code. `TRUST_SKLEARN_ARTIFACTS=true` is an explicit
process-local development opt-in, not a sandbox or provenance check. Use private,
trusted registry/storage data only; production defaults remain false. The seed
never loads unrelated artifacts. The server trust flag applies to all sklearn
artifacts it is asked to load, so do not expose this development server or storage
to untrusted writers. Matching training and serving dependencies are required.

After restart, persisted RUNNING status alone does not prove readiness. FastAPI
startup creates its container and automatically runs
`DeploymentManager.reconcile()` before accepting requests. The live script's deployment health request
invokes the existing targeted readiness check in the **server process**, reloading
the demo runtime if evicted; subsequent predictions also check readiness and revisions.
Use this check after each restart before treating the demo as ready. The seed
performs its own targeted smoke check but its in-memory runtime does not survive
exit. No global reconciliation is invoked by the seed, since that would inspect
and potentially load unrelated deployments. The existing explicit global reconcile
procedure is reserved for a quiescent startup with a fully trusted registry.
No demo metadata or artifacts are deleted during cleanup.

## Phase 6 routing and canary releases

`ServiceContainer` now composes `RoutingManager`, `TrafficRouter`, and an injected
`RoutingStrategy` alongside the existing inference services. `WeightedRandomStrategy`
accepts an optional `random.Random` instance, validates candidate weights, and
selects according to their relative probabilities with a short RNG lock. It never
accesses repositories, runtimes, or global random state. Weighted allocations are
probabilistic: a small request sample need not match percentages exactly.

`SQLiteRoutingRepository` stores a detached `RoutingSnapshot` containing a
`RoutingConfig`, integer revision, and optional `restored_from`. All persisted
weights are validated through the Domain on read and write. Current weights and
their revision commit atomically with an immutable history record and normalized
version references. SQL triggers reject UPDATE/DELETE on historical entries and
references. Foreign keys prevent deletion of any referenced version, including
zero-weight and historical destinations. There is no automatic history cleanup.
History protects metadata references; it is **not** an artifact backup, retention
archive, or guarantee against filesystem loss. Keep artifacts and dependencies
available for future rollback.

The numbered migration `app/infrastructure/database/migrations.py` installs the
three routing tables, immutability triggers, and a `schema_migrations` ledger entry
(version 6) in one transaction. Legacy model/version/deployment tables remain
unchanged. `initialize_database()` invokes this migration before serving traffic;
it does not use `create_all()` to evolve the routing schema. Repeated initialization
checks the ledger. Unexpected untracked routing tables fail rather than being
silently adopted. Perform initialization during quiescent startup; concurrent
startup writers may require an operator retry. This is a focused additive migration,
not a general migration framework.

### Eligibility, readiness, and admission

`RoutingManager` verifies the model, every referenced version, and the complete
Domain weight invariant (nonempty, integer 0..100, sum 100). Positive destinations
must be REGISTERED and have at least one RUNNING deployment for that exact version.
They must pass the existing `DeploymentManager.check_health()` load-readiness
contract; no generic smoke prediction is invented. Deployment revisions are
rechecked after readiness work. Zero-weight destinations must exist but need no
running deployment and may be INVALID, FAILED, or STOPPED; they are never selected.
Logical replicas retain the existing Phase 5 meaning and do not create workers.

`TrafficRouter` reads one current configuration snapshot, verifies eligible
metadata for **every** positive destination, rechecks deployment revisions, and
calls the strategy once. If any active destination is unavailable, the whole
automatic request fails with 503. It never drops a target and renormalizes remaining
weights. Operators must explicitly change the allocation or roll back. The router
does not load models or execute predictions. It cannot discover a latent artifact
or runtime failure in an unselected version: configuration checks prove readiness
at that time, and inference revalidates the selected runtime at admission. Existing
health checks can record FAILED when a runtime subsequently fails readiness.

`InferenceService` uses the selected version through its existing input validation,
runtime retrieval, readiness, final deployment revision check, vectorized prediction,
JSON normalization, and latency measurement. Explicit versions bypass the router
entirely, even if the routing configuration is missing or unavailable. Omitted or
null `version` uses routing; an empty or whitespace-only string remains invalid.
The response always identifies the version actually executed. Batch routing occurs
once per request, with one vectorized prediction preserving sample order.

Routing validation, routing persistence, selection, and inference admission do not
share a cross-repository transaction or distributed lease. A deployment stopped
before the final observed revision check is rejected. A stop after validation can
race a routing commit; later selection/admission rechecks reject the unavailable
allocation without fallback. A routing update after selection applies to subsequent
selections; the already selected request may use its earlier configuration snapshot.
Requests admitted before undeploy may finish. No database transaction or routing lock
spans prediction, and existing runtime caching and prediction locks are unchanged.

### Local routing management API

These controls have **no authentication or authorization** and must not be exposed
publicly. They use synchronous handlers in the existing finite FastAPI thread pool.
Existing body, tensor, and batch limits still apply.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/routing/{model_name}` | Read current weights, revision, timestamp, restore source |
| PUT | `/routing/{model_name}` | Validate and create/update a complete allocation |
| GET | `/routing/{model_name}/history` | Read immutable snapshots in ascending revision order |
| POST | `/routing/{model_name}/rollback` | Validate historical targets and append restored weights |

Create an initial canary with `PUT /routing/demo-linear`:

```json
{"weights":{"v1":90,"v2":10},"expected_revision":0}
```

Revision 0 is a create-only sentinel; initial stored revision is 1. The response
has this shape (timestamp varies):

```json
{"model_name":"demo-linear","weights":{"v1":90,"v2":10},"revision":1,"updated_at":"2026-10-20T00:00:00Z","restored_from":null}
```

Update to `75/25`, `50/50`, or `0/100` by sending the entire replacement weights
and the last observed `expected_revision`. Each successful mutation adds exactly
one revision, even when weights are unchanged. Revision data is mandatory; stale
writers and duplicate initial creation return 409. Invalid or duplicate JSON weight
entries return 422; unknown models/versions/current configurations/history return
404. Unavailable destinations or supporting services return sanitized 503 errors.
No database or artifact paths are included in HTTP errors.

Automatic prediction uses the existing endpoints:

```json
{"inputs":[[4.0],[5.0]]}
```

For `demo-linear/v1` the predictions are approximately `[9.0,11.0]`; for `v2` they
are `[13.0,16.0]`. An explicit `"version":"v1"` continues to give only v1 results.

To restore revision 1 when the current revision is 3, send
`POST /routing/demo-linear/rollback`:

```json
{"target_revision":1,"expected_revision":3}
```

The manager checks current versions and readiness again, and the repository
atomically appends revision 4 with restored weights and `restored_from: 1`.
Revisions 2 and 3 remain unchanged. An unavailable historical destination blocks
rollback; rollback is never an unchecked history-pointer change.

### Optional persistent v2 and demonstrations

Keep the same project working directory, `DATABASE_URL`, and `ARTIFACT_STORAGE_DIR`
for the seed and server. The original Phase 5 seeder and live checks remain valid.
To opt into a second trusted, locally trained version and initial 90/10 allocation:

```powershell
$env:TRUST_SKLEARN_ARTIFACTS = 'true'
python -m examples.seed_development_model
python -m examples.seed_routing_models
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

The focused routing seeder requires existing matching v1 metadata/bytes and a
RUNNING v1 deployment. It never overwrites the v1 artifact or restarts v1. It
verifies exact freshly generated bytes before deserializing either version, creates
missing v2 as `y=3*x+1`, and deploys it as `demo-api-v2-deployment`. Inactive v2
IDs remain unchanged; replacements use `-2`, `-3`, etc., and ready replacements
are reused. Configuration collisions fail explicitly. An initial routing allocation
is created only after readiness checks; existing operator routing stays unchanged.
Run one seeder at a time with trusted private storage. Pickle trust restrictions
and dependency-version mismatch behavior remain unchanged.

The existing live checks still test explicit v1 predictions without mutating routing:

```powershell
python -m examples.test_live_api
```

After the optional v2 setup, add `--routing` to these live checks to read the
current allocation and verify an automatic v1/v2 prediction without changing it:
`python -m examples.test_live_api --routing`.

Use Swagger `/docs` for optional-version predictions and routing controls. No
external server is needed for the deterministic routing demonstration:

```powershell
python -m examples.routing_demo
python -m pytest -q
```

The demo uses temporary SQLite/artifacts, two real sklearn models, a seeded RNG,
HTTP requests, canary counts, an allocation change, immutable history, rollback,
a stale-update conflict, and strict rejection after undeploy. Tests additionally
cover separate repository writers, injected partial failures, and admission races.
There is no sticky routing, automated canary analysis, rollout scheduler,
authentication infrastructure, distributed locking, or production observability stack.

## Phase 7 monitoring

The application-owned `ServiceContainer` injects `MetricsCollector` into
`InferenceService`. The collector depends on `InferenceLogRepository`, implemented
by `SQLiteInferenceLogRepository` using the existing session factory. Monitoring
routes access the collector, never SQLAlchemy sessions. There are no shared
mutable request observations, in-memory log buffers, background workers, or
external monitoring services. The original prediction response fields, routing
behavior, runtime cache, prediction locks, and deployment admission rules remain
unchanged. Directly constructed inference services may omit the optional collector
for compatibility; application-container services have monitoring enabled.

### Observation identity, privacy, and failures

The single observation boundary is `InferenceService.predict()`. `batch_predict()`
delegates to it, so a batch is **one request**, with one vectorized prediction and
at most one finalized observation. Explicit and automatically routed attempts use
the same boundary. Successful inference, in-service validation failures, missing
models/versions, routing failures, unavailable deployments, revision conflicts,
loading failures, execution failures, and output normalization failures are observed.
HTTP/Pydantic failures rejected before entering the service (including malformed
JSON and oversized bodies), unknown HTTP paths, health checks, routing-management
operations, and monitoring reads are excluded.

Each finalized attempt receives a fresh UUID observation `request_id`, unrelated
to model versions or client-provided HTTP headers. It is internal log identity,
not a transport correlation ID, and is not added to prediction responses. SQLite
enforces uniqueness: duplicate append raises `DuplicateInferenceLogError` and
preserves the existing record. No automatic write retry is performed.

`InferenceLog` keeps its constructor argument order and legacy `error_message`
field. Failed observations may now explicitly use `version=None` if routing has
not selected a version, and `model_name=None` if identifier validation failed.
Non-null identifiers must remain nonempty; successful observations still require
both identifiers. Explicit version attempts retain their requested version even
if registry lookup later fails. These narrowly scoped nullable fields avoid fake
version/model identifiers. Existing UTC and finite nonnegative latency validation
remain intact.

Application observations use `error_message` exclusively for an allowlisted,
bounded code. The SQL column and HTTP field are named `error_code`. Examples are
`invalid_input`, `not_found`, `routing_not_found`, `routing_unavailable`,
`deployment_conflict`, `unavailable`, `runtime_loading_failed`,
`artifact_unavailable`, `prediction_failed`, `serialization_failed`,
`model_contract_invalid`, `persistence_unavailable`, `unsupported_framework`, and
`internal_error`. Raw exception messages and stack traces are never persisted in
observations. Repository writes reject unrecognized error strings. No raw model
inputs, prediction outputs, serialized artifacts, credentials, or payload hashes
are generated/stored by inference monitoring. The preexisting optional `input_hash`
Domain field is retained for compatibility but application observations leave it
null, and monitoring DTOs do not expose it.

Monitoring is **best effort**. The service attempts a synchronous, short append
after finalizing inference, outside deployment/prediction transactions. A write
failure cannot change a successful result or mask the original inference exception.
It emits only the sanitized ERROR-level diagnostic `monitoring_write_failed`, without
backend text, payloads, or exception traces. Monitoring read failures return the
normal sanitized service error response, never fake zero metrics. Failed writes
may lose observations; an uncertain commit may already have persisted a record.
A process crash, termination, or BaseException interruption may leave no observation.
This is not exactly-once durable delivery or a distributed event pipeline.

### Metric definitions

All calculations use **recorded, finalized service attempts** whose completion
UTC timestamp falls in the selected half-open interval `[start, end)`, with the
specified model, version, and optional success filters. A missing version filter
includes known and unknown versions; `unknown_version=true` selects only null
versions and cannot be combined with `version`. Per-version results include a
null-version group when applicable. Querying a model with no observations returns
zero metrics rather than requiring that model to currently exist in the registry.

| Field | Definition |
| --- | --- |
| `total_requests` | Count of recorded attempts matching the filters |
| `successful_requests` | Count that completed prediction and output normalization/schema validation |
| `failed_requests` | Total minus successes, including expected user errors and unavailable versions |
| `error_rate` | Failures / total; 0.0 for zero observations |
| `throughput_requests_per_second` | Total / selected window duration in seconds |
| `average_latency_ms` | Arithmetic mean over successful observations only |
| `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms` | Exact nearest-rank percentiles over successful observations only |

Latency uses `perf_counter_ns()` elapsed wall-clock time from service entry through
validation, routing (if used), registry checks, readiness/loading, admission,
prediction, and output normalization/validation. Failures measure through the point
of failure. It excludes the subsequent monitoring append, HTTP network transfer,
HTTP body/Pydantic parsing, and waiting for the HTTP worker thread to start. The
successful observation latency equals the existing prediction response latency.
The observation timestamp is recorded at finalization, not request arrival.

For percentile `p`, sort the `n` successful latencies ascending and choose the
1-based rank `ceil(p*n/100)`. For `[1,2]`, p50 is 1 and p95/p99 are 2. Empty or
failure-only latency sets produce JSON `null` for mean and percentiles; counts
and rates remain finite. Counts and exact percentile ranks are computed in SQL,
using window functions and aggregate queries inside one read transaction. Only
aggregate rows reach Python; no unbounded log-object list is loaded. The SQL mean
normalizes by the maximum latency before averaging to avoid summation overflow
for large finite values. Exact sorting can still be expensive for large windows.

With neither endpoint supplied, the default interval is the **last five minutes**.
If only `end` is supplied, the interval is the preceding five minutes; if only
`start` is supplied, `end` is now. Timestamps must be timezone-aware and are
normalized to UTC. Windows must have positive duration and span at most 31 days.
Throughput is an observed average over this whole window, including idle time,
not peak server capacity. These are inference observations, **not a performance
benchmark**: logging losses, process scheduling, cache state, and the chosen window
affect interpretation. No liveness or health-check traffic inflates request counts.

### Monitoring API and pagination

All endpoints are local-development-only without authentication/authorization.
They use the existing synchronous FastAPI thread pool and sanitized error handlers.

| Method | Endpoint | Response |
| --- | --- | --- |
| GET | `/metrics` | Aggregate across matching observations |
| GET | `/metrics/{model_name}` | Aggregate for the path model |
| GET | `/metrics/{model_name}/versions` | Version-ordered aggregates, null version first |
| GET | `/inference-logs` | Bounded sanitized log page and next cursor |

Filters are `start`, `end` (aware ISO timestamps), `model_name`, `version`,
`unknown_version`, and `success`. On model-path endpoints, the path model takes
precedence over the optional query model. Version aggregation returns at most 200
groups; requests exceeding that cardinality return 422 and require narrower filters.
Empty version aggregates are `[]`. Empty ungrouped metrics contain zero counts,
zero rates, and null latencies. For example:

```json
{"model_name":"demo-linear","version":null,"start":"2026-10-20T00:00:00Z","end":"2026-10-20T00:05:00Z","total_requests":0,"successful_requests":0,"failed_requests":0,"error_rate":0.0,"throughput_requests_per_second":0.0,"average_latency_ms":null,"p50_latency_ms":null,"p95_latency_ms":null,"p99_latency_ms":null}
```

Log parameters additionally include `limit` (default 50, range 1..200) and `cursor`
(default 0, nonnegative insertion ID). Logs are ordered by insertion ID ascending,
which is stable even for concurrent writes or equal/out-of-order timestamps.
`next_cursor` is the final returned insertion ID only when another matching row
exists, otherwise null. Empty pages contain `logs: []`. Example response:

```json
{"logs":[{"request_id":"observation-uuid","model_name":"demo-linear","version":null,"latency_ms":0.5,"success":false,"error_code":"routing_not_found","timestamp":"2026-10-20T00:01:00Z"}],"next_cursor":null,"start":"2026-10-20T00:00:00Z","end":"2026-10-20T00:05:00Z"}
```

Reuse the returned `start`/`end` and all filters when following `next_cursor`.
Pagination is an append-only feed, not a frozen export: newly appended rows with
matching timestamps may appear on later pages. A fresh default window on each
page changes the query. No raw SQL IDs, error messages, inputs, or outputs are
exposed in log records; the cursor is the opaque continuation value to reuse.

Examples in Swagger `/docs` or a browser:

```text
/metrics/demo-linear?version=v1
/metrics/demo-linear/versions
/inference-logs?model_name=demo-linear&success=false&limit=20
```

### Migration, ownership, and verification

Migration **7** adds only `inference_logs` and indexes for completion time and
model/version/time filtering. It reuses explicit UTC storage and unique request IDs.
`initialize_database()` applies numbered migrations 6 and 7 during normal container
startup. It does not use `create_all()` to migrate monitoring tables. Run startup
quiescently and back up development data before schema upgrades. Unexpected
untracked tables fail explicitly; DDL and the migration ledger entry commit
atomically. Tests simulate an interrupted migration and verify successful retry.
Existing models, versions, deployment revisions, immutable routing history, and
artifact bytes are preserved. Logs intentionally have no registry foreign keys:
failed attempts may refer to missing versions, and observations neither retain
artifacts nor prevent registry deletion. No retention/purge service is implemented.

No extra connection, session, executor, or cache is globally owned by monitoring.
Each repository operation uses the injected session factory with its own explicit
transaction; application shutdown disposes the container's existing engine. Reads
see committed records in the selected SQLite file, not a production-wide,
consistent distributed snapshot. SQLite writer contention can delay appends or
cause best-effort loss; no cross-process delivery or aggregation guarantee is claimed.

```powershell
python -m pytest -q
python -m examples.monitoring_demo
python -m examples.test_live_api --monitoring
# With optional v2 routing already seeded:
python -m examples.test_live_api --routing --monitoring
```

The monitoring demo creates temporary SQLite/artifacts, deploys two trusted sklearn
versions, configures routing, executes explicit and routed/batched inference,
records a controlled failure, reads logs and aggregate/per-version metrics, checks
repository/container restart persistence, and removes temporary data. Existing
persistent v1/v2 seeders and routing configurations are unchanged. Live verification
checks metrics/logs optionally; default Phase 5/6 checks remain compatible. Its
prediction requests naturally add observations but never overwrite artifacts or
change routing. No automated test writes to persistent development storage.

Phase 8 hardening, Prometheus/Grafana, tracing, OpenTelemetry, brokers, authentication,
automated canary decisions, autoscaling, and long-term archival remain out of scope.

## Phase 7.5 recovery and resource safety

### Startup ownership and recovery policy

The FastAPI lifespan constructs the application-owned `ServiceContainer`, performs
existing migrations, and runs the existing `DeploymentManager.reconcile()` in a
worker thread **before startup completes or HTTP dependencies expose services**.
Caller-injected containers also reconcile; their cleanup remains caller-owned.
Overlapping lifespans on the same application instance are rejected. Concurrent
startup calls sharing one container join one in-progress reconciliation through a
future; its short bookkeeping lock is released before database/model work. Each
subsequent sequential lifespan revalidates again, rather than trusting an old
report. Explicit quiescent `manager.reconcile()` remains available to Python users.

This is a **single application process / single lifecycle owner** MVP. Run:

```powershell
$env:TRUST_SKLEARN_ARTIFACTS = 'true' # Only for explicitly trusted sklearn storage
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Multiple independent Uvicorn workers, separate containers sharing lifecycle state,
and concurrent seeding/deployment during startup are unsupported. The in-process
application/container guards are not distributed synchronization and cannot detect
or arbitrate every separately launched process. Stop other lifecycle writers before
starting or explicitly reconciling the application. No distributed lease is claimed.

Recovery checks current model/version registration, REGISTERED status, and artifact
existence even when a caller-injected container already has a cached runtime.
RUNNING deployments then use the existing load-readiness check. Fresh application
containers have empty caches, so startup actually reads and deserializes their active
versions; this is **not metadata-only recovery**. Normal startup does not invent
prediction smoke inputs. Load readiness proves a loaded object with predict capability,
not accuracy for every input, worker-process readiness, or external serving capacity.
Logical replicas remain metadata; no independent workers or CPU/RAM quotas exist.

| Situation | Startup policy |
| --- | --- |
| RUNNING, registered and available runtime | Revalidate; preserve RUNNING and its revision if unchanged |
| Interrupted DEPLOYING | Compare-and-swap to FAILED; do not restart |
| FAILED / STOPPED | Leave inactive; do not load historical models |
| Missing/corrupt/oversized artifact, INVALID version, disabled sklearn trust | Record per-deployment failure, detach unavailable cached references where applicable, persist FAILED under revision guard, continue with other deployments |
| Revision conflict / concurrent stop | Report isolated conflict; never overwrite newer lifecycle state |
| Database open/migration/list failure, repository infrastructure failure, or non-conflict failure recording recovery state | Abort startup; owned resources close; no serving readiness is advertised |

The API may start with some deployments unavailable. `/health` remains process
liveness and does not claim all models are ready. The startup report contains explicit
health observations and failures internally; it is not returned through a public API.
Diagnostics use sanitized error codes and aggregate counts, never artifact paths,
backend messages, or traces. Enable INFO logging for `app.core.container` to see
`startup_reconciliation_complete` counts; deployment recovery failures emit ERROR
`startup_deployment_recovery_failed`, and fatal startup emits ERROR
`application_startup_failed`. Fatal ASGI startup exceptions suppress backend exception
rendering. A completion count is not a claim that every deployment recovered.

**Trust remains disabled by default.** Starting against persisted sklearn deployments
without `TRUST_SKLEARN_ARTIFACTS=true` fails their loading and records FAILED; there is
no silent trust override. Only enable this setting for private, explicitly trusted
artifacts. Pickle remains capable of arbitrary code execution; byte limits are not a
sandbox. After correcting a failure, create a new deployment through existing Python
services rather than forcing FAILED or STOPPED back to RUNNING. No automatic repair
of operator data, artifact overwrite, rollout, or recovery retry is performed.

Startup scans deployment metadata and loads active versions sequentially. Cost scales
with persisted records and loading costs; startup may be slow for many large models.
The bounded LRU remains active throughout recovery, so successfully checked versions
can be evicted before startup finishes. They need not all remain cached. Later requests
reload on demand and repeat readiness and fresh deployment-revision admission checks.
Reports are point-in-time observations, not a durable execution lease. Existing strict
routing failure policy and best-effort inference monitoring remain unchanged.

### Runtime artifact byte limit

`MAX_RUNTIME_ARTIFACT_BYTES` defaults to **100 MiB** and must be a strictly positive
integer. It is separate from the reserved upload limit. The container injects it into
`ModelRuntimeManager`; direct callers can set `max_artifact_bytes`. Both sklearn and
ONNX use the same guard before loader selection/deserialization.

The narrow `ArtifactStorage.retrieve_limited(key, max_bytes=...)` extension preserves
ordinary registry save/download/delete contracts. Local storage opens the validated
artifact, checks the opened descriptor's size **before reading content**, then reads
bounded chunks with at most one extra byte to detect growth. Races or understated
file sizes cannot bypass the bounded read. The runtime manager independently checks
returned bytes/type/length, so a faulty backend cannot feed oversized bytes to a loader.
Exactly-at-limit artifacts are allowed. `ArtifactTooLargeError` is sanitized and uses
the existing artifact-unavailable HTTP semantics (503). Artifacts and version metadata
are not deleted or changed on rejection; deployment recovery may record FAILED.

The base storage method provides a retrieve-then-check compatibility fallback for old
adapters. It cannot bound memory before their retrieval: custom adapters must override
it with safe pre-read/bounded behavior for the same protection as local storage.
Changing a limit applies to new loads; it does not retrospectively resize an already
loaded runtime. Registered artifacts are still assumed immutable and privately owned.

This is a serialized-byte limit, **not a RAM limit**. Retrieved bytes, temporary copies,
expanded estimators, native ONNX allocations, and predictions can consume much more
memory. There is no per-model memory estimator, hard process limit, load timeout, or
process isolation. In-flight loads for distinct versions are not globally byte-limited.

### Cache and process memory visibility

The existing cache remains bounded by **entry count**, with LRU eviction, per-key
single-flight loading, independent key concurrency, and generation-safe invalidation.
Recovery preflight failure detaches an unavailable cached runtime; references already
held by callers remain usable. Eviction/invalidation never closes a runtime in use and
does not promise immediate reclamation by Python or native libraries.

`GET /runtime/resources` is a **local-development-only** operational endpoint, not an
inference request or a deployment-readiness check. It returns:

```json
{"rss_bytes":195567616,"rss_available":true,"cache":{"hits":2,"misses":1,"waits":0,"loads":1,"load_failures":0,"evictions":0,"current_size":1,"max_size":16,"in_flight":0}}
```

Cache counters retain their existing lifetime semantics; clear does not reset them.
`current_size` is completed cached runtimes, not resident bytes; `in_flight` counts
tracked loads and excludes detached generations still held by callers. No model names,
artifact paths, process IDs, command lines, or environment settings are exposed.
This unauthenticated endpoint must not be exposed publicly.

The injected `RuntimeResourceMonitor` uses a small standard-library platform probe:
Windows reports current working-set bytes through `GetProcessMemoryInfo`; Linux reads
resident pages from `/proc/self/statm` and multiplies by the system page size. Unsupported
platforms or unavailable OS counters return `rss_bytes: null, rss_available: false`.
Linux requires readable mounted procfs. Windows is verified natively in this environment;
Linux parsing and unavailable-counter behavior are tested with deterministic fixtures.
RSS is an approximate whole-process observation, including shared/native pages according
to OS accounting, not peak RSS, per-model attribution, or enforced memory capacity.

### Recovery demonstration and verification

```powershell
python -m examples.recovery_demo
python -m pytest -q
```

The recovery demo uses temporary SQLite/artifacts, shuts down and recreates an owned
application, verifies automatic recovery, runs a prediction, displays cache/RSS data,
rejects a separate oversized artifact before deserialization, verifies artifact retention,
and closes resources. Tests cover partial failures, trust settings, interrupted startup,
revision conflicts, repeated/injected lifespans, readiness gating, metadata preservation,
bounded reads during growth, faulty backends, and the unchanged LRU/single-flight races.
Existing persistent development seeders, model artifacts, and operator routing remain
unchanged. No new database schema migration is required for Phase 7.5.

Phase 8 packaging, process workers/isolation, real replicas, autoscaling, distributed
locking, memory-aware eviction, cloud infrastructure, and hard CPU/RAM limits are not
implemented.
