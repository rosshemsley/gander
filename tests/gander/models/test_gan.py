from gander.models.gan import _interpolate_factors


def test_interpolate_factors():
    v = _interpolate_factors(0,10,1)
    assert v == [1.0]

    v = _interpolate_factors(0,10,2)
    assert v == [1.0, 0.0]

    v = _interpolate_factors(9,10,2)
    assert v == [0.55, 0.45]

    v = _interpolate_factors(0,10,3)
    assert v == [0.5, 0.5, 0.0]

    v = _interpolate_factors(9,10,3)
    assert v == [0.35, 0.35, 0.3]
