from datetime import datetime, timedelta, timezone

import pytest

from app.core.exceptions import DomainValidationError
from app.domain.model import Model


def test_valid_model_has_utc_timestamps():
    model = Model("classifier", "A classifier")
    assert model.name == "classifier"
    assert model.description == "A classifier"
    assert model.created_at.tzinfo is timezone.utc
    assert model.updated_at.tzinfo is timezone.utc


@pytest.mark.parametrize("name", ["", " ", "\t\n", None, 12])
def test_invalid_name(name):
    with pytest.raises(DomainValidationError):
        Model(name)


def test_aware_timestamps_are_normalized_to_utc():
    local = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    model = Model("classifier", created_at=local, updated_at=local)
    assert model.created_at.hour == 10
    assert model.created_at == local
    assert model.created_at.tzinfo is timezone.utc
    assert model.updated_at.tzinfo is timezone.utc


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_naive_timestamps_are_rejected(field):
    with pytest.raises(DomainValidationError):
        Model("classifier", **{field: datetime(2026, 1, 1)})
