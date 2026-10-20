-- Reference captured from the complete pre-bootstrap ORM plus migrations 6/7.
-- Ledger intentionally excluded; this is a schema equivalence fixture, not an upgrade.
CREATE INDEX ix_deployments_model_name ON deployments (model_name);

CREATE INDEX ix_inference_logs_model_time ON inference_logs (model_name, timestamp);

CREATE INDEX ix_inference_logs_model_version_time ON inference_logs (model_name, version, timestamp);

CREATE INDEX ix_inference_logs_time ON inference_logs (timestamp);

CREATE TABLE deployments (
	id VARCHAR NOT NULL,
	model_name VARCHAR NOT NULL,
	version VARCHAR NOT NULL,
	replicas INTEGER NOT NULL,
	cpu FLOAT NOT NULL,
	memory_mb INTEGER NOT NULL,
	status VARCHAR NOT NULL,
	endpoint VARCHAR,
	created_at VARCHAR(40) NOT NULL,
	updated_at VARCHAR(40) NOT NULL,
	revision INTEGER NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT fk_deployments_model_version FOREIGN KEY(model_name, version) REFERENCES model_versions (model_name, version) ON DELETE RESTRICT,
	CONSTRAINT ck_deployments_replicas CHECK (replicas >= 0),
	CONSTRAINT ck_deployments_resources CHECK (cpu > 0 AND memory_mb > 0),
	CONSTRAINT ck_deployments_revision CHECK (revision >= 0)
);

CREATE TABLE inference_logs (
	id INTEGER NOT NULL,
	request_id VARCHAR NOT NULL,
	model_name VARCHAR,
	version VARCHAR,
	latency_ms FLOAT NOT NULL,
	success BOOLEAN NOT NULL,
	error_code VARCHAR(40),
	timestamp VARCHAR(40) NOT NULL,
	input_hash VARCHAR,
	PRIMARY KEY (id),
	CHECK (latency_ms >= 0),
	CHECK (success IN (0, 1)),
	CHECK (success = 0 OR (model_name IS NOT NULL AND version IS NOT NULL)),
	UNIQUE (request_id)
);

CREATE TABLE model_versions (
	id INTEGER NOT NULL,
	model_name VARCHAR NOT NULL,
	version VARCHAR NOT NULL,
	artifact_location VARCHAR NOT NULL,
	framework VARCHAR NOT NULL,
	input_schema JSON NOT NULL,
	output_schema JSON NOT NULL,
	status VARCHAR NOT NULL,
	created_at VARCHAR(40) NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT uq_model_versions_identity UNIQUE (model_name, version),
	FOREIGN KEY(model_name) REFERENCES models (name) ON DELETE RESTRICT
);

CREATE TABLE models (
	id INTEGER NOT NULL,
	name VARCHAR NOT NULL,
	description VARCHAR NOT NULL,
	created_at VARCHAR(40) NOT NULL,
	updated_at VARCHAR(40) NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT uq_models_name UNIQUE (name)
);

CREATE TABLE routing_current (
	model_name VARCHAR NOT NULL,
	weights JSON NOT NULL,
	revision INTEGER NOT NULL,
	updated_at VARCHAR(40) NOT NULL,
	restored_from INTEGER,
	PRIMARY KEY (model_name),
	CHECK (revision >= 1),
	FOREIGN KEY(model_name) REFERENCES models (name) ON DELETE RESTRICT
);

CREATE TABLE routing_history (
	model_name VARCHAR NOT NULL,
	revision INTEGER NOT NULL,
	weights JSON NOT NULL,
	updated_at VARCHAR(40) NOT NULL,
	restored_from INTEGER,
	PRIMARY KEY (model_name, revision),
	CHECK (revision >= 1),
	FOREIGN KEY(model_name) REFERENCES models (name) ON DELETE RESTRICT
);

CREATE TABLE routing_version_references (
	model_name VARCHAR NOT NULL,
	revision INTEGER NOT NULL,
	version VARCHAR NOT NULL,
	PRIMARY KEY (model_name, revision, version),
	FOREIGN KEY(model_name, revision) REFERENCES routing_history (model_name, revision) ON DELETE RESTRICT,
	FOREIGN KEY(model_name, version) REFERENCES model_versions (model_name, version) ON DELETE RESTRICT
);

CREATE TRIGGER routing_history_no_delete BEFORE DELETE ON routing_history BEGIN SELECT RAISE(ABORT, 'Routing history is immutable'); END;

CREATE TRIGGER routing_history_no_update BEFORE UPDATE ON routing_history BEGIN SELECT RAISE(ABORT, 'Routing history is immutable'); END;

CREATE TRIGGER routing_version_references_no_delete BEFORE DELETE ON routing_version_references BEGIN SELECT RAISE(ABORT, 'Routing history is immutable'); END;

CREATE TRIGGER routing_version_references_no_update BEFORE UPDATE ON routing_version_references BEGIN SELECT RAISE(ABORT, 'Routing history is immutable'); END;
