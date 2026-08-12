import numpy as np

from dml_health_benchmark.features import rich_dictionary_dimension, rich_nonlinear_dictionary


def test_rich_dictionary_dimension_and_shape():
    X = np.arange(30, dtype=float).reshape(5, 6) / 10.0
    Z = rich_nonlinear_dictionary(X, rho=0.3)
    assert Z.shape == (5, rich_dictionary_dimension(6))
    assert Z.shape[1] == 29
    assert np.allclose(Z[:, :6], X)
