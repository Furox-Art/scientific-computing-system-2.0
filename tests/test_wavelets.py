"""Tests for cds2.wavelets."""

import numpy as np
import pytest

from cds2 import wavelets


class TestHaarDWT:
    def test_known_example_normalized(self) -> None:
        approx, detail = wavelets.haar_dwt([1.0, 2.0, 3.0, 4.0], normalize=True)
        assert np.allclose(approx, np.array([3.0, 7.0]) / np.sqrt(2.0), rtol=0, atol=1e-12)
        assert np.allclose(detail, np.array([-1.0, -1.0]) / np.sqrt(2.0), rtol=0, atol=1e-12)

    def test_known_example_unnormalized(self) -> None:
        approx, detail = wavelets.haar_dwt([1.0, 2.0, 3.0, 4.0], normalize=False)
        assert list(approx) == [1.5, 3.5]
        assert list(detail) == [-0.5, -0.5]

    def test_constant_signal_details_zero(self) -> None:
        for normalize in (True, False):
            approx, detail = wavelets.haar_dwt(np.ones(8), normalize=normalize)
            assert np.allclose(detail, 0.0, atol=1e-12)
            expected = np.sqrt(2.0) if normalize else 1.0
            assert np.allclose(approx, expected, atol=1e-12)

    def test_odd_length_raises(self) -> None:
        with pytest.raises(ValueError, match="signal length must be even"):
            wavelets.haar_dwt([1.0, 2.0, 3.0])

    def test_single_point_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two points"):
            wavelets.haar_dwt([1.0])

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D series"):
            wavelets.haar_dwt([[1.0, 2.0], [3.0, 4.0]])


class TestHaarIDWT:
    def test_roundtrip_normalized(self) -> None:
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(64)
        approx, detail = wavelets.haar_dwt(signal, normalize=True)
        restored = wavelets.haar_idwt(approx, detail, normalize=True)
        assert np.allclose(restored, signal, rtol=0, atol=1e-12)

    def test_roundtrip_unnormalized(self) -> None:
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(32)
        approx, detail = wavelets.haar_dwt(signal, normalize=False)
        restored = wavelets.haar_idwt(approx, detail, normalize=False)
        assert np.allclose(restored, signal, rtol=0, atol=1e-12)

    def test_known_example(self) -> None:
        restored = wavelets.haar_idwt([1.5, 3.5], [-0.5, -0.5], normalize=False)
        assert list(restored) == [1.0, 2.0, 3.0, 4.0]

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            wavelets.haar_idwt([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            wavelets.haar_idwt([], [])

    def test_non_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 1-D"):
            wavelets.haar_idwt([[1.0]], [[1.0]])


class TestDWTLevels:
    def test_level_shapes(self) -> None:
        result = wavelets.dwt_levels(np.arange(16.0), levels=3)
        assert result.original_length == 16
        assert [a.size for a in result.approximations] == [8, 4, 2]
        assert [d.size for d in result.details] == [8, 4, 2]

    def test_first_level_matches_haar_dwt(self) -> None:
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        result = wavelets.dwt_levels(signal, levels=2)
        approx, detail = wavelets.haar_dwt(signal)
        assert np.allclose(result.approximations[0], approx, atol=1e-12)
        assert np.allclose(result.details[0], detail, atol=1e-12)

    def test_constant_signal_all_details_zero(self) -> None:
        result = wavelets.dwt_levels(np.ones(8), levels=3)
        assert all(np.allclose(d, 0.0, atol=1e-12) for d in result.details)

    def test_frozen_result(self) -> None:
        result = wavelets.dwt_levels([1.0, 2.0], levels=1)
        with pytest.raises(AttributeError):
            result.original_length = 99

    def test_levels_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="levels must be at least 1"):
            wavelets.dwt_levels([1.0, 2.0], levels=0)

    def test_negative_levels_raise(self) -> None:
        with pytest.raises(ValueError, match="levels must be at least 1"):
            wavelets.dwt_levels([1.0, 2.0], levels=-2)

    def test_signal_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="signal too short"):
            wavelets.dwt_levels([1.0, 2.0], levels=3)

    def test_single_point_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            wavelets.dwt_levels([1.0, 2.0, 3.0, 4.0], levels=10)


class TestIDWTLevels:
    @pytest.mark.parametrize("normalize", [True, False])
    def test_roundtrip_random_multilevel(self, normalize: bool) -> None:
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(48)
        decomposition = wavelets.dwt_levels(signal, levels=3, normalize=normalize)
        restored = wavelets.idwt_levels(decomposition, normalize=normalize)
        assert restored.size == decomposition.original_length == 48
        assert np.allclose(restored, signal, rtol=0, atol=1e-12)

    def test_two_levels_hand_computed(self) -> None:
        decomposition = wavelets.dwt_levels([1.0, 2.0, 3.0, 4.0], levels=2)
        assert np.allclose(decomposition.approximations[-1], 10.0 / 2.0, atol=1e-12)
        assert list(wavelets.idwt_levels(decomposition)) == pytest.approx([1.0, 2.0, 3.0, 4.0])


class TestWaveletDenoise:
    @pytest.fixture()
    def noisy_signal(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 4.0 * np.pi, 256)
        smooth = np.sin(t) + 0.5 * np.cos(0.5 * t)
        rng = np.random.default_rng(42)
        return smooth, smooth + rng.normal(0.0, 0.1, size=t.size)

    def test_reduces_noise_error(self, noisy_signal: tuple[np.ndarray, np.ndarray]) -> None:
        smooth, noisy = noisy_signal
        result = wavelets.wavelet_denoise(noisy)
        before = float(np.linalg.norm(noisy - smooth))
        after = float(np.linalg.norm(result.data - smooth))
        assert result.thresholded_counts > 0
        assert after < before

    def test_metadata_and_shape(self, noisy_signal: tuple[np.ndarray, np.ndarray]) -> None:
        _smooth, noisy = noisy_signal
        result = wavelets.wavelet_denoise(noisy, threshold_factor=2.0, levels=3)
        assert result.data.shape == noisy.shape
        assert result.levels == 3

    def test_constant_signal_nothing_thresholded(self) -> None:
        result = wavelets.wavelet_denoise(np.ones(16))
        assert result.thresholded_counts == 0
        assert np.allclose(result.data, 1.0, atol=1e-12)

    def test_unnormalized_path(self, noisy_signal: tuple[np.ndarray, np.ndarray]) -> None:
        smooth, noisy = noisy_signal
        result = wavelets.wavelet_denoise(noisy, levels=2, normalize=False)
        before = float(np.linalg.norm(noisy - smooth))
        after = float(np.linalg.norm(result.data - smooth))
        assert result.thresholded_counts > 0
        assert after < before
