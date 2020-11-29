"""Unit tests for core.py."""

import pytest
import numpy as np
from scipy.stats import entropy
from entrogrammer import core
from entrogrammer import classifier


def test_type_error():
    """Test that an error is raised if invalid input is given."""
    with pytest.raises(TypeError):
        core.global_entropy('invalid')


def test_notclassified_error():
    """Test that an error is raised if data has not been classified."""
    C = classifier.BinaryClassifier(np.zeros(2), 1)
    with pytest.raises(ValueError):
        core.global_entropy(C)


def test_badbase_error():
    """Test that an error is raised if base is invalid."""
    C = classifier.BinaryClassifier(np.zeros(2), 1)
    C.classify()
    with pytest.raises(TypeError):
        core.global_entropy(C, 4)


def test_entropy_zero():
    """Test case where entropy is 0."""
    C = classifier.BinaryClassifier(np.zeros(2), 1)
    C.classify()
    HG = core.global_entropy(C)
    assert HG == 0


def test_binary_entropy():
    """Test case where two options are equiprobable.

    This is a known case that should equal 1 bit of information from:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    C = classifier.BinaryClassifier(np.array([0, 1]), 0.5)
    C.classify()
    HG = core.global_entropy(C, 2)
    assert HG == 1


def test_1D_entrogram():
    """Test 1D entrogram."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    HR, win_size = core.calculate_entrogram(C)
    # for this simple case we can calculate expected values
    assert win_size == [2, 3]
    assert HR[0] == entropy((0.5, 0.5)) / entropy((1/3, 2/3))
    assert HR[1] == 1.0


def test_1D_entrogram_windows():
    """Test 1D entrogram with window parameters."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    HR, win_size = core.calculate_entrogram(C, min_win=2, max_win=3)
    # for this simple case we can calculate expected values
    assert win_size == [2, 3]
    assert HR[0] == entropy((0.5, 0.5)) / entropy((1/3, 2/3))
    assert HR[1] == 1.0


def test_1D_entrogram_invalid_minwin():
    """Test 1D entrogram with invalid min_win."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    with pytest.raises(ValueError):
        core.calculate_entrogram(C, min_win='bad', max_win=3)


def test_1D_entrogram_invalid_maxwin():
    """Test 1D entrogram with invalid max_win."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    with pytest.raises(ValueError):
        core.calculate_entrogram(C, min_win=2, max_win='bad')


def test_local_entropy_1D():
    """Test 1D local entropy calculation.

    This is a known case that should equal 1 bit of information from:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    C = classifier.BinaryClassifier(np.array([0, 1]), 0.5)
    C.classify()
    HL = core.local_entropy(C, 2, 2)
    assert np.all(HL == 1)


def test_local_entropy_1D_subset():
    """Test 1D local entropy calculation as portion of longer array."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    HL = core.local_entropy(C, 2)
    assert np.all(HL == entropy((0.5, 0.5)))


def test_local_entropy_1D_tuple():
    """Test 1D local entropy calculation with tuple input."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    HL = core.local_entropy(C, (2,))
    assert np.all(HL == entropy((0.5, 0.5)))


def test_local_entropy_1D_tuple_float():
    """Test 1D local entropy calculation with tuple float input."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    HL = core.local_entropy(C, (2.0,))
    assert np.all(HL == entropy((0.5, 0.5)))


def test_local_entropy_1D_bad_win():
    """Test 1D local entropy calculation with invalid tuple type."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    with pytest.raises(TypeError):
        core.local_entropy(C, ('invalid',))


def test_local_entropy_wrongwintype():
    """Test 1D local entropy calculation with invalid tuple type."""
    C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)
    C.classify()
    with pytest.raises(TypeError):
        core.local_entropy(C, 'invalid')


@pytest.mark.xfail(raises=NotImplementedError)
def test_local_entropy_2D_tuple():
    """Test 2D local entropy calculation with tuple."""
    C = classifier.BinaryClassifier(np.zeros((2, 2)), 0.5)
    C.classify()
    HL = core.local_entropy(C, (2, 2))
    assert HL == 0
