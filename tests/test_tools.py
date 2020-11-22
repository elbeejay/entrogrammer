"""Unit tests for tools.py."""

import pytest
import numpy as np
from entrogrammer import tools


def test_HL_1D_base2():
    """Test for 1-D local entropy with base 2."""
    data = np.zeros((100,))
    win_size = 10
    h = tools.HL_1D_base2(data, win_size)
    assert h == 0


def test_HL_1D_base10():
    """Test for 1-D local entropy with base 10."""
    data = np.zeros((100,))
    win_size = 10
    h = tools.HL_1D_base10(data, win_size)
    assert h == 0


def test_HL_1D_basee():
    """Test for 1-D local entropy with base e."""
    data = np.zeros((100,))
    win_size = 10
    h = tools.HL_1D_basee(data, win_size)
    assert h == 0


def test_unique_counts():
    """Test numba unique counts implementation."""
    arr = np.array([0, 0, 0, 1, 1, 2])
    cnts = tools.np_unique_impl(arr)
    assert cnts[0] == 3
    assert cnts[1] == 2
    assert cnts[2] == 1
