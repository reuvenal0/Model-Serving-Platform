"""Small shared validation helpers for domain construction."""

from datetime import datetime, timezone
from enum import Enum
from math import isfinite

from app.core.exceptions import DomainValidationError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise DomainValidationError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{field_name} must be a nonempty string")


def enum_value(value: Enum, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise DomainValidationError(f"{field_name} must be a {enum_type.__name__}")


def finite_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainValidationError(f"{field_name} must be a finite number")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise DomainValidationError(f"{field_name} must be a finite number")
