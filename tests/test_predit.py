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
def test_linear_regr(
    coefs: Dict[str, float],
    x: Dict[str, float],
    expected: float
) -> None:
    regr = skp.LinearRegression(coefs)
    y = regr.predict(x)
    assert isclose(y, expected)
