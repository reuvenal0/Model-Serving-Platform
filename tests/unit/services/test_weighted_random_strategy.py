"""Deterministic weighted selection independent of deployments and execution."""

from collections import Counter
from random import Random
from unittest.mock import Mock

import pytest

from app.core.exceptions import DomainValidationError
from app.services.weighted_random_strategy import WeightedRandomStrategy


@pytest.mark.parametrize('ticket,expected', [(0, 'v1'), (89, 'v1'), (90, 'v2'), (99, 'v2')])
def test_ticket_boundaries(ticket, expected):
    rng = Mock(spec=Random)
    rng.randrange.return_value = ticket
    assert WeightedRandomStrategy(rng).select({'v0': 0, 'v1': 90, 'v2': 10}) == expected
    rng.randrange.assert_called_once_with(100)


@pytest.mark.parametrize('weights', [{}, {'v1': 0}, {'v1': -1}, {'v1': 101}, {'v1': True},
                                    {'v1': 1.0}, {' ': 100}, {'v1': '100'}, None, []])
def test_invalid_candidates(weights):
    with pytest.raises(DomainValidationError):
        WeightedRandomStrategy(Random(1)).select(weights)


def test_distribution_and_reproducibility():
    first, second = WeightedRandomStrategy(Random(42)), WeightedRandomStrategy(Random(42))
    a = [first.select({'v1': 90, 'v2': 10, 'off': 0}) for _ in range(10000)]
    b = [second.select({'v1': 90, 'v2': 10, 'off': 0}) for _ in range(10000)]
    assert a == b
    assert 850 < Counter(a)['v2'] < 1150
    assert 'off' not in a


def test_relative_weights_need_not_sum_to_100():
    strategy = WeightedRandomStrategy(Random(7))
    counts = Counter(strategy.select({'a': 3, 'b': 1}) for _ in range(4000))
    assert 2800 < counts['a'] < 3200


def test_single_destination_is_stable():
    strategy = WeightedRandomStrategy(Random(1))
    assert {strategy.select({'v1': 100, 'v2': 0}) for _ in range(100)} == {'v1'}
