"""Tests for cds2.image."""

import numpy as np
import pytest

from cds2 import image


class TestConvolve2d:
    def test_identity_kernel_returns_copy(self) -> None:
        img = np.arange(12.0, dtype=float).reshape(3, 4)
        out = image.convolve2d(img, [[1.0]])
        np.testing.assert_array_equal(out, img)
        assert out is not img

    def test_same_and_valid_shapes(self) -> None:
        img = np.zeros((6, 6))
        kernel = np.ones((3, 3))
        assert image.convolve2d(img, kernel, mode="same").shape == (6, 6)
        assert image.convolve2d(img, kernel, mode="valid").shape == (4, 4)

    def test_impulse_response_is_flipped_kernel_footprint(self) -> None:
        img = np.zeros((6, 6))
        img[1, 1] = 1.0
        kernel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        out = image.convolve2d(img, kernel)
        expected = np.zeros((6, 6))
        expected[0:3, 0:3] = kernel
        np.testing.assert_array_equal(out, expected)

    def test_convolution_differs_from_correlation_on_asymmetric_input(self) -> None:
        img = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        kernel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        out = image.convolve2d(img, kernel)
        assert out[1, 2] == 1.0
        assert out[1, 0] == 0.0
        assert int(np.count_nonzero(out)) == 1

    def test_valid_mode_interior_only(self) -> None:
        img = np.full((5, 5), 2.0)
        kernel = np.array([[0.25]])
        out = image.convolve2d(img, kernel, mode="valid")
        np.testing.assert_allclose(out, np.full((5, 5), 0.5))

    def test_rejects_bad_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be same or valid"):
            image.convolve2d(np.zeros((4, 4)), [[1.0]], mode="full")

    def test_rejects_even_kernel(self) -> None:
        with pytest.raises(ValueError, match="square with an odd side"):
            image.convolve2d(np.zeros((4, 4)), np.ones((2, 2)))

    def test_rejects_nonsquare_kernel(self) -> None:
        with pytest.raises(ValueError, match="square with an odd side"):
            image.convolve2d(np.zeros((4, 4)), np.ones((1, 3)))

    def test_rejects_non_2d_image(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D"):
            image.convolve2d([1.0, 2.0], [[1.0]])

    def test_rejects_empty_image(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D"):
            image.convolve2d(np.empty((0, 0)), [[1.0]])


class TestGaussianKernel:
    def test_normalized_and_symmetric(self) -> None:
        kernel = image.gaussian_kernel(size=5, sigma=1.0)
        assert kernel.sum() == pytest.approx(1.0)
        np.testing.assert_array_equal(kernel, kernel.T)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1])

    def test_center_is_maximum(self) -> None:
        kernel = image.gaussian_kernel(size=7, sigma=1.5)
        center = kernel[3, 3]
        assert center == kernel.max()
        assert center > kernel[0, 0]

    def test_sigma_scales_spread(self) -> None:
        small = image.gaussian_kernel(sigma=0.5)
        large = image.gaussian_kernel(sigma=2.0)
        assert large[2, 2] > large[0, 0] > small[0, 0]

    def test_rejects_sigma_not_positive(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            image.gaussian_kernel(sigma=0.0)

    def test_rejects_negative_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            image.gaussian_kernel(sigma=-1.0)

    def test_rejects_even_size(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.gaussian_kernel(size=4)

    def test_rejects_small_size(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.gaussian_kernel(size=1)


class TestGaussianBlur:
    def test_blur_lowers_noise_variance(self) -> None:
        rng = np.random.default_rng(42)
        noisy = rng.normal(0.0, 1.0, size=(32, 32))
        blurred = image.gaussian_blur(noisy, sigma=2.0, size=7)
        interior = (slice(4, -4), slice(4, -4))
        assert blurred[interior].std() < noisy.std()

    def test_blur_preserves_constant_interior(self) -> None:
        img = np.full((10, 10), 3.5)
        blurred = image.gaussian_blur(img)
        interior = blurred[2:-2, 2:-2]
        assert interior.mean() == pytest.approx(3.5)
        np.testing.assert_allclose(interior, interior[0, 0])

    def test_blur_propagates_validation(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.gaussian_blur(np.zeros((8, 8)), size=2)


class TestSobelEdges:
    def test_vertical_step_gradient(self) -> None:
        step = np.zeros((8, 8))
        step[:, 4:] = 1.0
        result = image.sobel_edges(step)
        assert isinstance(result, image.EdgeResult)
        magnitude = result.magnitude
        edge_strength = magnitude[1:-1, 2:4].max()
        flat_strength = magnitude[1:-1, 0].max()
        assert edge_strength == pytest.approx(4.0)
        assert flat_strength == 0.0

    def test_magnitude_spot_check(self) -> None:
        step = np.zeros((6, 6))
        step[:, 3:] = 1.0
        magnitude = image.sobel_edges(step).magnitude
        for row in (1, 2, 3, 4):
            assert magnitude[row, 2] == pytest.approx(4.0)
            assert magnitude[row, 3] == pytest.approx(4.0)

    def test_direction_on_vertical_step(self) -> None:
        step = np.zeros((6, 6))
        step[:, 3:] = 1.0
        direction = image.sobel_edges(step).direction
        assert direction[2, 2] == pytest.approx(0.0)
        assert direction[2, 3] == pytest.approx(0.0)

    def test_flat_image_has_no_edges(self) -> None:
        flat = np.full((6, 6), 2.0)
        magnitude = image.sobel_edges(flat).magnitude
        assert magnitude[1:-1, 1:-1].max() == pytest.approx(0.0)

    def test_shapes_match_input(self) -> None:
        img = np.arange(36.0).reshape(6, 6) / 35.0
        result = image.sobel_edges(img)
        assert result.magnitude.shape == (6, 6)
        assert result.direction.shape == (6, 6)

    def test_rejects_non_2d_image(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D"):
            image.sobel_edges([1.0, 2.0, 3.0])


class TestDownsample:
    def test_mean_blocks(self) -> None:
        img = np.arange(16.0).reshape(4, 4)
        out = image.downsample(img, factor=2, method="mean")
        expected = np.array([[2.5, 4.5], [10.5, 12.5]])
        np.testing.assert_allclose(out, expected)

    def test_max_blocks(self) -> None:
        img = np.arange(16.0).reshape(4, 4)
        out = image.downsample(img, factor=2, method="max")
        expected = np.array([[5.0, 7.0], [13.0, 15.0]])
        np.testing.assert_allclose(out, expected)

    def test_min_blocks(self) -> None:
        img = np.arange(16.0).reshape(4, 4)
        out = image.downsample(img, factor=2, method="min")
        expected = np.array([[0.0, 2.0], [8.0, 10.0]])
        np.testing.assert_allclose(out, expected)

    def test_crops_to_multiple_of_factor(self) -> None:
        img = np.ones((6, 7))
        out = image.downsample(img, factor=4)
        assert out.shape == (1, 1)
        assert out[0, 0] == pytest.approx(1.0)

    def test_factor_three(self) -> None:
        img = np.ones((6, 6)) * 4.0
        out = image.downsample(img, factor=3)
        assert out.shape == (2, 2)
        assert out[0, 0] == pytest.approx(4.0)

    def test_rejects_bad_method(self) -> None:
        with pytest.raises(ValueError, match="method must be mean, max or min"):
            image.downsample(np.ones((4, 4)), method="median")

    def test_rejects_small_factor(self) -> None:
        with pytest.raises(ValueError, match="factor must be at least 2"):
            image.downsample(np.ones((4, 4)), factor=1)

    def test_rejects_bad_image(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D"):
            image.downsample(np.empty((0, 4)))


class TestBinarize:
    def test_output_values(self) -> None:
        img = np.array([[0.1, 0.6], [0.5, 0.9]])
        out = image.binarize(img, threshold=0.5)
        expected = np.array([[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_equal(out, expected)
        assert set(np.unique(out)) <= {0.0, 1.0}

    def test_high_threshold_gives_zeros(self) -> None:
        out = image.binarize(np.ones((3, 3)), threshold=2.0)
        assert not out.any()

    def test_rejects_bad_image(self) -> None:
        with pytest.raises(ValueError, match="non-empty 2-D"):
            image.binarize([1, 2, 3], threshold=0.5)


class _MorphologyBase:
    @staticmethod
    def bright_square() -> np.ndarray:
        binary = np.zeros((10, 10))
        binary[3:7, 3:7] = 1.0
        return binary


class TestErode(_MorphologyBase):
    def test_shrinks_bright_square(self) -> None:
        eroded = image.erode(self.bright_square())
        assert int(np.count_nonzero(eroded)) == 4
        assert eroded[4, 4] == 1.0
        assert eroded[3, 3] == 0.0

    def test_all_ones_stays_full(self) -> None:
        eroded = image.erode(np.ones((6, 6)))
        np.testing.assert_array_equal(eroded, np.ones((6, 6)))

    def test_structure_size_five(self) -> None:
        eroded = image.erode(self.bright_square(), structure_size=5)
        assert int(np.count_nonzero(eroded)) == 0

    def test_rejects_non_binary(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            image.erode(np.full((4, 4), 0.5))

    def test_rejects_even_structure_size(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.erode(np.ones((4, 4)), structure_size=2)

    def test_rejects_small_structure_size(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.erode(np.ones((4, 4)), structure_size=1)


class TestDilate(_MorphologyBase):
    def test_grows_bright_square(self) -> None:
        dilated = image.dilate(self.bright_square())
        assert int(np.count_nonzero(dilated)) == 36
        assert dilated[2, 5] == 1.0
        assert dilated[1, 1] == 0.0

    def test_all_zeros_stays_empty(self) -> None:
        dilated = image.dilate(np.zeros((6, 6)))
        assert not dilated.any()

    def test_structure_size_five(self) -> None:
        dilated = image.dilate(self.bright_square(), structure_size=5)
        assert int(np.count_nonzero(dilated)) == 64

    def test_roundtrip_never_shrinks(self) -> None:
        binary = self.bright_square()
        restored = image.erode(image.dilate(binary))
        assert int(np.count_nonzero(restored)) >= int(np.count_nonzero(binary))

    def test_rejects_non_binary(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            image.dilate(np.array([[0.0, 2.0], [1.0, 0.0]]))

    def test_rejects_even_structure_size(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.dilate(np.zeros((4, 4)), structure_size=4)

    def test_rejects_small_structure_size(self) -> None:
        with pytest.raises(ValueError, match="odd and at least 3"):
            image.dilate(np.zeros((4, 4)), structure_size=0)
