"""Public renderer contracts checked against independent, scalar expectations."""

import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from flattener import FlattenerError, RenderConfig, Status, render


def _grid(height, width):
    """Derive the reference coordinates without production coordinate helpers."""
    result = np.empty((height, width, 2), dtype=np.float32)
    for row in range(height):
        for column in range(width):
            result[row, column] = (
                2 * (column + 0.5) / width - 1,
                2 * (row + 0.5) / height - 1,
            )
    return result


def _pattern(height, width):
    rows, columns = np.indices((height, width))
    return np.stack(
        [
            (37 * rows + 13 * columns) % 256,
            (19 * rows + 41 * columns + 17) % 256,
            (11 * rows + 7 * columns + 139) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def _scalar_expected(image, coordinates, target=None, fill=(255, 255, 255)):
    """Scalar bilinear border oracle; no torch or production helpers involved."""
    height, width = image.shape[:2]
    output = np.empty((*coordinates.shape[:2], 3), dtype=np.uint8)
    for row, column in np.ndindex(coordinates.shape[:2]):
        gx, gy = map(float, coordinates[row, column])
        active = target is None or bool(target[row, column])
        if not active or not (-1 <= gx <= 1 and -1 <= gy <= 1):
            output[row, column] = fill
            continue
        x = min(max((gx + 1) * width / 2 - 0.5, 0), width - 1)
        y = min(max((gy + 1) * height / 2 - 0.5, 0), height - 1)
        x0, y0 = math.floor(x), math.floor(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        ax, ay = x - x0, y - y0
        for channel in range(3):
            value = (
                (1 - ax) * (1 - ay) * int(image[y0, x0, channel])
                + ax * (1 - ay) * int(image[y0, x1, channel])
                + (1 - ax) * ay * int(image[y1, x0, channel])
                + ax * ay * int(image[y1, x1, channel])
            )
            output[row, column, channel] = math.floor(value + 0.5)
    return output


def _assert_success(result, output_size):
    assert result.status == Status.SUCCESS
    assert result.reason_code is None
    assert result.used_fallback is False
    assert result.image.dtype == np.uint8
    assert result.image.shape == (*output_size, 3)
    for mask in (result.source_mask, result.target_mask, result.valid_mask):
        assert mask.dtype == np.bool_
        assert mask.shape == output_size


def test_nonsquare_identity_keeps_pixels_axes_and_channels():
    image = _pattern(3, 5)
    result = render(image, _grid(3, 5))

    _assert_success(result, (3, 5))
    np.testing.assert_array_equal(result.image, image)
    assert result.source_mask.all()
    assert result.target_mask.all()
    assert result.valid_mask.all()


@pytest.mark.parametrize("dx,dy", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_integer_source_displacement_moves_content_in_the_opposite_direction(dx, dy):
    image = _pattern(4, 4)
    coordinates = _grid(4, 4)
    coordinates += np.array([2 * dx / 4, 2 * dy / 4], dtype=np.float32)
    fill = (9, 23, 81)
    expected = np.full_like(image, fill)
    expected_valid = np.zeros((4, 4), dtype=bool)
    for row, column in np.ndindex((4, 4)):
        if 0 <= row + dy < 4 and 0 <= column + dx < 4:
            expected[row, column] = image[row + dy, column + dx]
            expected_valid[row, column] = True

    result = render(image, coordinates, config=RenderConfig(fill_rgb=fill))

    _assert_success(result, (4, 4))
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_array_equal(result.valid_mask, expected_valid)
    assert result.metadata["validation"]["out_of_bounds_fraction"] == 0.25


@pytest.mark.parametrize("dx,dy", [(0.5, 0), (0, 0.5), (0.5, 0.5)])
def test_exact_half_pixel_means(dx, dy):
    rows, columns = np.indices((4, 4))
    base = 16 * rows + 4 * columns
    image = np.stack([base, base + 64, base + 128], axis=-1).astype(np.uint8)
    coordinates = _grid(4, 4)
    coordinates += np.array([2 * dx / 4, 2 * dy / 4], dtype=np.float32)

    result = render(image, coordinates)

    _assert_success(result, (4, 4))
    np.testing.assert_array_equal(result.image, _scalar_expected(image, coordinates))
    assert result.valid_mask.all()


def test_quantization_rounds_exact_half_values_up():
    image = np.array(
        [[[0, 2, 250], [1, 3, 249], [4, 6, 244], [9, 11, 235]]],
        dtype=np.uint8,
    )
    coordinates = np.array([[[-0.5, 0], [0, 0], [0.5, 0], [1, 0]]], dtype=np.float32)
    expected = np.array(
        [[[1, 3, 250], [3, 5, 247], [7, 9, 240], [9, 11, 235]]],
        dtype=np.uint8,
    )

    result = render(image, coordinates)

    np.testing.assert_array_equal(result.image, expected)


@pytest.mark.parametrize(
    "source_size,output_size",
    [((3, 5), (7, 8)), ((7, 8), (3, 5)), ((4, 7), (5, 3)), ((5, 3), (2, 8))],
)
def test_irregular_resizing_matches_independent_bilinear_oracle(source_size, output_size):
    image = _pattern(*source_size)
    coordinates = _grid(*output_size)

    result = render(image, coordinates)

    _assert_success(result, output_size)
    expected = _scalar_expected(image, coordinates)
    difference = np.abs(result.image.astype(np.int16) - expected.astype(np.int16))
    assert difference.max() <= 1
    assert result.valid_mask.all()


def test_physical_boundaries_are_inclusive_and_outside_coordinates_are_filled():
    image = np.array([[[10, 20, 30], [200, 210, 220]]], dtype=np.uint8)
    coordinates = np.array(
        [[[-1.0001, 0], [-1, 0], [1, 0], [1.0001, 0], [0, -1], [0, 1], [0, 1.0001]]],
        dtype=np.float32,
    )
    fill = (1, 2, 3)

    result = render(
        image,
        coordinates,
        config=RenderConfig(max_out_of_bounds_fraction=1, fill_rgb=fill),
    )

    expected_mask = [[False, True, True, False, True, True, False]]
    np.testing.assert_array_equal(result.source_mask, expected_mask)
    np.testing.assert_array_equal(result.valid_mask, expected_mask)
    np.testing.assert_array_equal(
        result.image,
        [[fill, [10, 20, 30], [200, 210, 220], fill, [105, 115, 125], [105, 115, 125], fill]],
    )


def test_source_target_and_valid_masks_are_separate_and_denominator_is_target_count():
    image = _pattern(2, 2)
    coordinates = np.array([[[-0.5, -0.5], [2, 0], [0.5, 0.5], [2, 2]]], dtype=np.float32)
    target = np.array([[True, True, False, False]])
    fill = (4, 5, 6)
    result = render(
        image,
        coordinates,
        target_mask=target,
        config=RenderConfig(max_out_of_bounds_fraction=0.5, fill_rgb=fill),
    )

    np.testing.assert_array_equal(result.source_mask, [[True, False, True, False]])
    np.testing.assert_array_equal(result.target_mask, target)
    np.testing.assert_array_equal(result.valid_mask, [[True, False, False, False]])
    np.testing.assert_array_equal(result.image, [[image[0, 0], fill, fill, fill]])
    validation = result.metadata["validation"]
    assert validation["target_pixels"] == 2
    assert validation["valid_pixels"] == 1
    assert validation["out_of_bounds_pixels"] == 1
    assert validation["out_of_bounds_fraction"] == 0.5
    assert validation["fold_check"] == "not_applicable"

    failure = render(
        image,
        coordinates,
        target_mask=target,
        config=RenderConfig(max_out_of_bounds_fraction=0.49, on_error="return"),
    )
    assert failure.status == Status.INVALID_TRANSFORM
    assert failure.reason_code == "excessive_out_of_bounds"


@pytest.mark.parametrize(
    "value,policy", [(np.nan, "raise"), (np.inf, "return"), (-np.inf, "original")]
)
def test_active_nonfinite_coordinates_fail_with_each_policy(value, policy):
    image = _pattern(2, 3)
    coordinates = _grid(4, 5)
    coordinates[1, 2, 0] = value
    before_image = image.copy()
    before_coordinates = coordinates.copy()
    config = RenderConfig(on_error=policy)

    if policy == "raise":
        with pytest.raises(FlattenerError) as caught:
            render(image, coordinates, config=config)
        assert caught.value.status == Status.INVALID_TRANSFORM
        assert caught.value.reason_code == "non_finite_grid"
        assert caught.value.metadata["requested_output_size"] == [4, 5]
    else:
        result = render(image, coordinates, config=config)
        assert result.status == Status.INVALID_TRANSFORM
        assert result.reason_code == "non_finite_grid"
        assert result.source_mask is result.target_mask is result.valid_mask is None
        assert result.metadata["source_size"] == [2, 3]
        assert result.metadata["requested_output_size"] == [4, 5]
        assert result.metadata["timings"]["render"] is None
        if policy == "original":
            np.testing.assert_array_equal(result.image, image)
            assert result.used_fallback is True
            assert result.metadata["actual_output_size"] == [2, 3]
            assert not np.shares_memory(result.image, image)
        else:
            assert result.image is None
            assert result.used_fallback is False
            assert result.metadata["actual_output_size"] is None
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(coordinates, before_coordinates)


def test_invalid_failure_policy_always_raises_input_error():
    with pytest.raises(FlattenerError) as caught:
        render(_pattern(2, 3), _grid(2, 3), config=RenderConfig(on_error="ignore"))

    assert caught.value.status == Status.INPUT_ERROR


def test_fold_failure_remains_failure_in_original_fallback():
    image = _pattern(3, 5)
    reflected = _grid(2, 4)
    reflected[..., 0] *= -1
    result = render(image, reflected, config=RenderConfig(on_error="original"))

    assert result.status == Status.INVALID_TRANSFORM
    assert result.reason_code == "fold_or_degenerate"
    assert result.used_fallback is True
    assert result.metadata["validation"]["fold_check"] == "failed"
    assert result.metadata["validation"]["failed_cells"] == 3
    assert result.metadata["requested_output_size"] == [2, 4]
    assert result.metadata["actual_output_size"] == [3, 5]
    np.testing.assert_array_equal(result.image, image)


def test_success_metadata_records_configuration_sizes_validation_and_real_stages():
    config = RenderConfig(fill_rgb=(1, 2, 3), max_output_pixels=100)
    result = render(_pattern(3, 5), _grid(4, 6), config=config)
    metadata = result.metadata

    assert metadata["spec_version"] == "0.1"
    assert metadata["source_size"] == [3, 5]
    assert metadata["requested_output_size"] == [4, 6]
    assert metadata["actual_output_size"] == [4, 6]
    assert metadata["model_id"] is None
    assert metadata["model_sha256"] is None
    assert json.loads(json.dumps(metadata["config"])) == json.loads(json.dumps(asdict(config)))
    assert metadata["timing_scope"] == "render"
    timings = metadata["timings"]
    assert set(timings) == {
        "load_model",
        "prepare_input",
        "predict",
        "process_coordinates",
        "render",
        "save",
        "total",
    }
    for stage in ("load_model", "predict", "save"):
        assert timings[stage] is None
    for stage in ("prepare_input", "process_coordinates", "render", "total"):
        assert math.isfinite(timings[stage])
        assert timings[stage] >= 0
        assert timings["total"] >= timings[stage]
    assert isinstance(metadata["versions"], dict)
    assert metadata["versions"]
    assert isinstance(metadata["cpu_threads"], int)
    assert metadata["cpu_threads"] > 0
    validation = metadata["validation"]
    assert validation["target_pixels"] == validation["valid_pixels"] == 24
    assert validation["out_of_bounds_pixels"] == 0
    assert validation["out_of_bounds_fraction"] == 0
    assert validation["fold_check"] == "passed"
    assert validation["checked_cells"] == 15
    assert validation["excluded_cells"] == validation["failed_cells"] == 0
    assert validation["min_jacobian_ratio"] == pytest.approx(1, abs=1e-6)
    json.dumps(metadata, allow_nan=False)
