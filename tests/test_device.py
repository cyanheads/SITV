"""Tests for device management functions."""
import pytest
import torch
from unittest.mock import patch
from sitv.core.device import get_device_map


class TestDeviceMap:
    """Tests for get_device_map function."""

    @patch("torch.cuda.is_available")
    def test_cuda_available_returns_auto(self, mock_cuda):
        """When CUDA is available, should return 'auto'."""
        mock_cuda.return_value = True
        assert get_device_map() == "auto"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_mps_available_returns_none(self, mock_mps, mock_cuda):
        """When MPS is available (not CUDA), should return None."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        assert get_device_map() is None

    @patch("torch.cuda.is_available")
    def test_cpu_only_returns_none(self, mock_cuda):
        """When only CPU is available, should return None."""
        mock_cuda.return_value = False
        # MPS check will fail naturally if not on Mac
        assert get_device_map() is None
