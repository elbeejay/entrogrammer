"""Unit tests for tools.py."""

import pytest
import numpy as np
from scipy.stats import entropy
from entrogrammer import tools


def test_HL_1D_base2_proper():
    """Test that if for 1D with base 2 works."""
    HL = tools.calculate_HL(np.zeros((2,)), 2, 2)
    assert np.all(HL == 0)


def test_HL_1D_base10_proper():
    """Test that if for 1D with base 10 works."""
    HL = tools.calculate_HL(np.zeros((2,)), 2, 10)
    assert np.all(HL == 0)


def test_HL_1D_basee_proper():
    """Test that if for 1D with base e works."""
    HL = tools.calculate_HL(np.zeros((2,)), 2, np.e)
    assert np.all(HL == 0)


@pytest.mark.xfail(raises=NotImplementedError)
def test_HL_2D_base2_proper():
    """Test that if for 2D with base 2 works."""
    HL = tools.calculate_HL(np.zeros((2, 2)), 2, 2)
    assert np.all(HL == 0)


@pytest.mark.xfail(raises=NotImplementedError)
def test_HL_2D_base10_proper():
    """Test that if for 2D with base 10 works."""
    HL = tools.calculate_HL(np.zeros((2, 2)), 2, 10)
    assert np.all(HL == 0)


@pytest.mark.xfail(raises=NotImplementedError)
def test_HL_2D_basee_proper():
    """Test that if for 2D with base e works."""
    HL = tools.calculate_HL(np.zeros((2, 2)), 2, np.e)
    assert np.all(HL == 0)


@pytest.mark.xfail(raises=NotImplementedError)
def test_HL_3D_base2_proper():
    """Test that if for 3D with base 2 works."""
    HL = tools.calculate_HL(np.zeros((2, 2, 2)), 2, 2)
    assert np.all(HL == 0)


@pytest.mark.xfail(raises=NotImplementedError)
def test_HL_3D_base10_proper():
    """Test that if for 3D with base 10 works."""
    HL = tools.calculate_HL(np.zeros((2, 2, 2)), 2, 10)
    assert np.all(HL == 0)


@pytest.mark.xfail(raises=NotImplementedError)
def test_HL_3D_basee_proper():
    """Test that if for 3D with base e works."""
    HL = tools.calculate_HL(np.zeros((2, 2, 2)), 2, np.e)
    assert np.all(HL == 0)


def test_HL_excessive_dims():
    """Test with more than 3 dims."""
    with pytest.raises(TypeError):
        tools.calculate_HL(np.zeros((2, 2, 2, 2)), 2, np.e)


def test_HL_1D_base2():
    """Test for 1-D local entropy with base 2."""
    data = np.zeros((100,))
    win_size = 10
    h = np.zeros_like(data).astype('float')
    cnt = np.zeros_like(data)
    h = tools.HL_1D_base2(data, win_size, h, cnt)
    assert np.all(h == 0)


def test_HL_1D_noslide():
    """Test for 1-D local entropy with window==data."""
    data = np.zeros((10,))
    data[0] = 1
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    win_size = 10
    h = np.zeros_like(data).astype('float')
    cnt = np.zeros_like(data)
    h = tools.HL_1D_basee(data, win_size, h, cnt)
    assert np.all(h == entropy(probs))


def test_HL_1D_base10():
    """Test for 1-D local entropy with base 10."""
    data = np.zeros((100,))
    win_size = 10
    h = np.zeros_like(data).astype('float')
    cnt = np.zeros_like(data)
    h = tools.HL_1D_base10(data, win_size, h, cnt)
    assert np.all(h == 0)


def test_HL_1D_basee():
    """Test for 1-D local entropy with base e."""
    data = np.zeros((100,))
    win_size = 10
    h = np.zeros_like(data).astype('float')
    cnt = np.zeros_like(data)
    h = tools.HL_1D_basee(data, win_size, h, cnt)
    assert np.all(h == 0)


def test_unique_counts():
    """Test numba unique counts implementation."""
    arr = np.array([0, 0, 0, 1, 1, 2])
    cnts = tools.np_unique_impl(arr)
    assert cnts[0] == 3
    assert cnts[1] == 2
    assert cnts[2] == 1
