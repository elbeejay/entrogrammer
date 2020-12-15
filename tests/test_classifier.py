"""Unit tests for classifier.py."""

import pytest
import numpy as np
import xarray as xr
from entrogrammer import classifier

xr_data = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
np_data = np.zeros((2, 2))


def test_xarray_classifier():
    """Test that xarray data type can be handled."""
    C = classifier.BinaryClassifier(xr_data, 1)
    assert isinstance(C, classifier.BaseClassifier) is True
    assert type(C) == classifier.BinaryClassifier
    assert np.all(C.data == np.zeros((2, 2)))


def test_numpy_classifier():
    """Test that numpy data type can be handled."""
    C = classifier.BinaryClassifier(np_data, 1)
    assert isinstance(C, classifier.BaseClassifier) is True
    assert type(C) == classifier.BinaryClassifier
    assert np.all(C.data == np.zeros((2, 2)))


def test_typeerror_classifier():
    """Test that a type error will be raised."""
    with pytest.raises(TypeError):
        classifier.BinaryClassifier('invalid', 1)


def test_binary_neg_threshold():
    """Test binary classifier with a negative threshold."""
    vals = np.zeros((2, 2))
    vals[0, :] = -5
    vals[1, :] = -10
    # create classifier
    C = classifier.BinaryClassifier(vals, -7.5)
    assert np.all(C.data[0, :] == -5)
    assert np.all(C.data[1, :] == -10)
    assert np.all(C.classified[0, :] == 1)
    assert np.all(C.classified[1, :] == 0)
    # check threshold value
    assert C.threshold == -7.5
    # re-classify then check values
    C.classify(0)
    assert np.all(C.data[0, :] == -5)
    assert np.all(C.data[1, :] == -10)
    assert np.all(C.classified[0, :] == 0)
    assert np.all(C.classified[1, :] == 0)


def test_binary_pos_threshold():
    """Test binary classifier with a positive threshold."""
    vals = np.zeros((2, 2))
    vals[0, :] = 5
    vals[1, :] = 10
    # create classifier
    C = classifier.BinaryClassifier(vals, 7)
    assert np.all(C.data[0, :] == 5)
    assert np.all(C.data[1, :] == 10)
    assert np.all(C.classified[0, :] == 0)
    assert np.all(C.classified[1, :] == 1)


def test_binary_invalid_threshold():
    """Test binary classifier with invalid threshold."""
    with pytest.raises(TypeError):
        classifier.BinaryClassifier(np_data, 'invalid')


def test_override_threshold():
    """Test overriding threshold in classify()."""
    vals = np.zeros((2, 2))
    vals[0, :] = 5
    vals[1, :] = 10
    # create classifier and assert threshold
    C = classifier.BinaryClassifier(vals, 7)
    assert C.threshold == 7
    assert np.all(C.data[0, :] == 5)
    assert np.all(C.data[1, :] == 10)
    # classify with different value then check values
    C.classify(1)
    assert C.threshold == 1
    assert np.all(C.data[0, :] == 5)
    assert np.all(C.data[1, :] == 10)
    assert np.all(C.classified[0, :] == 1)
    assert np.all(C.classified[1, :] == 1)


# check for jenkspy - skips test if jenkspy is not present
_skip_jenks = 0
try:
    from jenkspy import JenksNaturalBreaks
except Exception:
    _skip_jenks = 1


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_int():
    """Test JenksClassifier with integer."""
    vals = np.zeros((5,))
    vals[3:] = 1
    # classify w/ Jenks
    C = classifier.JenksClassifier(vals, 3)
    # make assertion
    assert len(np.unique(C.classified)) == 2


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_int_2D():
    """Test JenksClassifier with integer."""
    vals = np.zeros((5, 5))
    vals[3:, :] = 1
    # classify w/ Jenks
    C = classifier.JenksClassifier(vals, 3)
    # make assertion
    assert len(np.unique(C.classified)) == 2
    assert C.nb_class == 3


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_float():
    """Test JenksClassifier with float."""
    vals = np.zeros((5,))
    vals[3:] = 1
    # classify w/ Jenks
    C = classifier.JenksClassifier(vals, 3.0)
    # make assertion
    assert len(np.unique(C.classified)) == 2
    assert C.nb_class == 3


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_invalid_nbclass_at_init():
    """Test JenksClassifier with invalid input."""
    vals = np.zeros((5,))
    vals[3:] = 1
    with pytest.raises(TypeError):
        classifier.JenksClassifier(vals, 'invalid')


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_invalid_nbclass_later_on():
    """Test JenksClassifier with invalid input."""
    vals = np.zeros((5,))
    vals[3:] = 1
    C = classifier.JenksClassifier(vals, 3)
    with pytest.raises(ValueError):
        C.classify('invalid')


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_small_nbclass():
    """Test JenksClassifier with too small nb_class input."""
    vals = np.zeros((5,))
    vals[3:] = 1
    with pytest.raises(ValueError):
        classifier.JenksClassifier(vals, 1)


@pytest.mark.skipif(_skip_jenks == 1, reason="jenkspy not available")
def test_jenks_big_nbclass():
    """Test JenksClassifier with too big nb_class input."""
    vals = np.zeros((5,))
    vals[3:] = 1
    with pytest.raises(ValueError):
        classifier.JenksClassifier(vals, 6)


# tests for histogram-like binning
def test_hist_int():
    """Test HistogramClassifier with integer."""
    vals = np.array([0, 10, 20, 100])
    C = classifier.HistogramClassifier(vals, 2)
    assert np.all(C.classified == np.array([1, 1, 1, 3]))


def test_hist_range():
    """Test HistogramClassifier with range."""
    vals = np.array([20, 40, 60])
    C = classifier.HistogramClassifier(vals, 2.0, range=(0, 100))
    assert np.all(C.classified == np.array([1, 1, 2]))
    assert C.range == (0, 100)


def test_hist_defaults():
    """Test HistogramClassifier with defaults."""
    vals = np.zeros((10,))
    vals[6:] = 5
    C = classifier.HistogramClassifier(vals)
    assert np.all(C.classified[:6] == 1)
    assert np.all(C.classified[6:] == 11)
    # re-apply classifier with new info
    C.classify(bins=2, range=(0, 6))
    assert np.all(C.classified[:6] == 1)
    assert np.all(C.classified[6:] == 2)
    assert C.bins == 10


def test_hist_badbin():
    """error with bad bin value"""
    vals = np.zeros((10,))
    with pytest.raises(TypeError):
        classifier.HistogramClassifier(vals, bins='badbin')


def test_hist_badrange():
    """error with bad range value"""
    vals = np.zeros((10,))
    with pytest.raises(TypeError):
        classifier.HistogramClassifier(vals, bins=10.0, range='badrange')
