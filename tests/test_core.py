"""Unit tests for core.py."""

import pytest
import numpy as np
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
