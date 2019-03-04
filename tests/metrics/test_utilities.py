from itertools import  product

import pytest

from vivarium_public_health.metrics.utilities import QueryString


@pytest.mark.parametrize('reference, test', product([QueryString(''), QueryString('abc')], [QueryString(''), '']))
def test_query_string_empty(reference, test):
    result = str(reference)
    assert reference + test == result
    assert reference + test == QueryString(result)
    assert isinstance(reference + test, QueryString)

    assert test + reference == result
    assert test + reference == QueryString(result)
    assert isinstance(test + reference, QueryString)

    reference += test
    assert reference == result
    assert reference == QueryString(result)
    assert isinstance(reference, QueryString)

    test += reference
    assert test == result
    assert test == QueryString(result)
    assert isinstance(test, QueryString)


@pytest.mark.parametrize('a, b', product([QueryString('a')], [QueryString('b'), 'b']))
def test_query_string(a, b):
    assert a + b == 'a and b'
    assert a + b == QueryString('a and b')
    assert isinstance(a + b, QueryString)

    assert b + a == 'b and a'
    assert b + a == QueryString('b and a')
    assert isinstance(b + a, QueryString)

    a += b
    assert a == 'a and b'
    assert a == QueryString('a and b')
    assert isinstance(a, QueryString)

    b += a
    assert b == 'b and a and b'
    assert b == QueryString('b and a and b')
    assert isinstance(b, QueryString)
