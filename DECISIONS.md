# Design Decisions

## 1. Model Loading Strategy

**Hybrid: lazy runtime loading with startup/deployment readiness checks.** Model
registration stores metadata and artifacts without deserializing them. A runtime
cache miss loads the requested immutable `(model_name, version)` through the
injected loader registry. Deployment readiness requires a real runtime load;
startup recovery revalidates persisted active deployments before serving requests.
This can load models eagerly during recovery, but does not pin every model in RAM.

`ModelRuntimeManager` uses a bounded LRU cache. A per-key shared future prevents
duplicate concurrent loads of the same tracked generation; different keys can
load concurrently. Slow storage/loading runs outside the cache bookkeeping lock.
Failed loads are not cached. Invalidation detaches in-flight generations so they
cannot repopulate the cache, while existing callers may finish safely.

Local artifact loading checks file size before reading, uses bounded reads and
independently verifies returned bytes before deserialization. These checks limit
serialized bytes, not expanded model memory. Eviction releases cache references;
it does not promise immediate memory reclamation. Cache occupancy and process RSS
are visible, but neither enforces a model memory budget.

The main demo uses self-contained ONNX artifacts. sklearn loading requires an
explicit trust opt-in before Pickle deserialization; there is no Pickle sandbox.
Management upload selects only ONNX, never the sklearn loader.

## 2. Request Routing

`TrafficRouter` separates availability checks from selection. `RoutingConfig`
requires nonempty versions and integer weights from 0 to 100 summing to 100.
`WeightedRandomStrategy` draws an integer ticket in the supplied total weight
and selects its cumulative interval. It has an injectable `Random` instance and
a lock around RNG access, enabling reproducible tests without global random state.
Zero-weight destinations are never selected; invalid/empty candidates are rejected.

Every positive-weight destination must be RUNNING and ready. If any active
destination becomes unavailable, automatic routing fails rather than silently
discarding it and renormalizing the remainder. Zero-weight destinations must be
registered, but need not be running. Explicit versions bypass routing completely.
Each batched request selects one version and makes one vectorized prediction call.

Configuration updates use expected-revision compare-and-swap. Current state and
immutable history are persisted atomically in one SQLite transaction. A stale
writer receives a conflict instead of overwriting another operator's change.

Inference rechecks selected-version readiness and deployment revision at admission.
Routing validation and admission are separate operations, not a distributed
transaction. A stop between selection and admission rejects the request with no
fallback. Already-admitted predictions may finish after a subsequent stop.
Configured weights describe probabilities, not exact small-sample counts.

## 3. Model Isolation

**There is no process isolation.** Loaded sklearn estimators and ONNX sessions run
inside the FastAPI process. Per-runtime prediction locks serialize calls to each
runtime; different runtimes can execute concurrently. Logical replicas share the
same cached runtime instead of creating independent workers.

The HTTP inference admission controller uses a dedicated bounded executor and a
strictly bounded waiting queue. Capacity covers routing, readiness/runtime loading,
prediction and output processing. Excess work receives a sanitized 503 response.
Cancelling an HTTP waiter does not release capacity while synchronous work remains
active. Queue timeout bounds waiting only; it cannot terminate native execution.
Shutdown drains submitted work before closing application-owned resources, so a
permanently stuck native call may delay shutdown.

`ResourceConfig` records requested CPU/memory and deployment replicas are logical
state. Neither enforces operating-system limits. Compose limits constrain the
container as a whole, not individual models. RSS observation and LRU entry counts
are not hard RAM limits. Management/readiness work is outside inference admission.

ONNX is also not a security sandbox: only trusted models should be registered.
A native crash or resource exhaustion can affect the entire API process.
API keys plus genuine loopback protect operations; prediction remains public
through the default host-loopback binding. This is not a public production service.

## 4. Deployment Safety

Deployment creation persists DEPLOYING, verifies model-version eligibility and
runtime load-readiness, then persists RUNNING under a revision guard. Failures
attempt to persist FAILED. Readiness means an executable runtime is available;
it does not prove prediction quality or independent worker/process health. The
manager supports explicit smoke inputs, but the health endpoint invents none.

Startup reconciles persisted RUNNING records and marks interrupted DEPLOYING
records FAILED. Per-deployment recovery failures can leave the API live with some
models unavailable; fatal persistence/infrastructure failures prevent startup.
Stopped deployments cannot simply transition back to RUNNING. Revision checks
prevent stale recovery or health results from restoring concurrently changed state.
Recovery ownership assumes one application process, not multiple coordinated workers.

Gradual rollout changes explicit routing allocations, such as 100/0, 90/10 and
50/50, after positive-weight targets pass readiness. **Rollback is manual**: the
operator selects a historical revision and supplies the current expected revision.
The manager revalidates historical destinations against current availability,
then appends a new revision containing the restored weights. It never rewrites
history. There is no error-triggered rollback, canary analysis or rollout scheduler.

Artifact creation and metadata persistence are separate operations. Registration
compensates metadata failures by deleting only its newly created artifact; double
failures are explicit. These are not crash-safe distributed transactions. Monitoring
is best-effort and cannot mask inference errors or turn a successful prediction
into a failure. Fresh SQLite bootstrap creates the final schema; historical schema
upgrades are intentionally unsupported and incompatible databases fail safely.

## 5. One Thing I Would Do Differently With More Time

Prioritize a separate model execution process with an explicit worker lifecycle.
It would provide a real boundary for native crashes, enforceable execution
deadlines and operating-system resource controls. The current `LoadedModel` and
loader responsibilities make a future execution boundary understandable, but no
worker abstraction or process-isolation guarantee is implemented today. This
would require new lifecycle/admission tests rather than relabeling logical replicas
as independent workers.
