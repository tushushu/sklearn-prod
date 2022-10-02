from typing import Dict
from math import isclose

import pytest
import sklearn_prod as skp


@pytest.mark.parametrize(
    'coefs, x, expected',
    [
        (
            {'foo': 0.0},
            {'foo': 0.0},
            0.0,
        ),
        (
            {'foo': 0.0, 'bar': 1.0, 'baz': 2.0},
            {'foo': 0.0, 'bar': 1.0, 'baz': 2.0},
            5.0,
        ),
    ],
)
def test_predict(
    coefs: Dict[str, float],
    x: Dict[str, float],
    expected: float
) -> None:
    regr = skp.LinearRegression(coefs)
    y = regr.predict(x)
    assert isclose(y, expected)


@pytest.mark.parametrize(
    'coefs, x, expected_error',
    [
        (
            {'foo': 0.0},
            {'bar': 0.0},
            KeyError,
        ),
        (
            {'foo': 0.0},
            {'foo': 0.0, 'bar': 0.0},
            ValueError,
        ),
        (
            {'foo': 0.0},
            {'bar': 0.0, 'baz': 0.0},
            ValueError,
        ),
        (
            dict(),
            {'foo': 0.0},
            ValueError,
        ),
        (
            {'foo': 0.0},
            dict(),
            ValueError,
        ),
        (
            {'foo': 0.0, 'bar': 0.0},
            {'foo': 0.0},
            ValueError,
        ),
        (
            {'foo': 0.0, 'bar': 0.0},
            {'baz': 0.0},
            ValueError,
        ),
    ],
)
def test_exceptions(
    coefs: Dict[str, float],
    x: Dict[str, float],
    expected_error: Exception,
) -> None:
    with pytest.raises(expected_error):
        regr = skp.LinearRegression(coefs)
        regr.predict(x)
