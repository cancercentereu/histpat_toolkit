import math

import numpy as np
import pytest

from histpat_toolkit.geom import (
    Circle,
    Ellipse,
    Point,
    Polygon,
    Rectangle,
    circle_mask,
    crop_rect_from_img,
    crop_shape_from_img,
    ellipse_mask,
    paste_rect_into_img,
    polygon_mask,
    rectangle_mask,
    rotate_vector,
    to_point,
    translate_point,
)


def test_point_creation_and_properties():
    p = Point(3, 4)
    assert isinstance(p, Point)
    assert p.x == 3
    assert p.y == 4


def test_point_equality():
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    p3 = Point(2, 3)
    assert p1 == p2
    assert p1 != p3


def test_point_operations():
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    assert (p1 + p2) == Point(4, 6)
    assert (p2 - p1) == Point(2, 2)
    assert (p1 * 2) == Point(2, 4)
    assert (p2 / 2) == Point(1.5, 2)
    assert (p2 // 2) == Point(1, 2)


def test_point_norm_and_dist():
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert p2.norm() == 5
    assert p1.dist(p2) == 5


def test_point_rotate():
    p = Point(1, 0)

    assert p.rotate(0) == p

    rotated = p.rotate(math.pi / 2)
    assert math.isclose(rotated.x, 0, abs_tol=1e-15)
    assert math.isclose(rotated.y, 1, abs_tol=1e-15)

    rotated = p.rotate(math.pi)
    assert math.isclose(rotated.x, -1, abs_tol=1e-15)
    assert math.isclose(rotated.y, 0, abs_tol=1e-15)

    rotated = p.rotate(3 * math.pi / 2)
    assert math.isclose(rotated.x, 0, abs_tol=1e-15)
    assert math.isclose(rotated.y, -1, abs_tol=1e-15)

    rotated = p.rotate(2 * np.pi)
    assert math.isclose(rotated.x, 1, abs_tol=1e-15)
    assert math.isclose(rotated.y, 0, abs_tol=1e-15)


def test_to_point():
    p = Point(1, 2)
    assert to_point(p) is p
    p2 = to_point((3, 4))
    assert isinstance(p2, Point)
    assert p2 == Point(3, 4)


def test_rotate_vector():
    vec = (1, 1)

    assert rotate_vector(vec, 0) == vec

    rotated = rotate_vector(vec, math.pi / 2)
    assert math.isclose(rotated.x, -1, abs_tol=1e-15)
    assert math.isclose(rotated.y, 1, abs_tol=1e-15)

    rotated = rotate_vector(vec, math.pi)
    assert math.isclose(rotated.x, -1, abs_tol=1e-15)
    assert math.isclose(rotated.y, -1, abs_tol=1e-15)

    rotated = rotate_vector(vec, 3 * math.pi / 2)
    assert math.isclose(rotated.x, 1, abs_tol=1e-15)
    assert math.isclose(rotated.y, -1, abs_tol=1e-15)

    rotated = rotate_vector(vec, 2 * math.pi)
    assert math.isclose(rotated.x, 1, abs_tol=1e-15)
    assert math.isclose(rotated.y, 1, abs_tol=1e-15)


def test_translate_point():
    point = (1, 1)
    vector = (2, 3)
    translated = translate_point(point, vector)
    assert translated == Point(3, 4)


def test_rectangle_with_center_and_scale():
    rect = Rectangle.with_center(5, 5, 4, 2, rot=0)
    assert math.isclose(rect.x + rect.w / 2, 5, abs_tol=1e-15)
    assert math.isclose(rect.y + rect.h / 2, 5, abs_tol=1e-15)

    scaled = rect.scale(2)
    assert scaled.w == rect.w * 2
    assert scaled.h == rect.h * 2


def test_rectangle_points():
    rect = Rectangle(0, 0, 2, 2, rot=0)
    pts = rect.points()
    expected_points = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)]
    for p, ep in zip(pts, expected_points):
        assert p == ep

    rect = Rectangle(0, 0, 4, 3, rot=math.pi / 2)
    pts = rect.points()
    expected_points = [Point(0, 0), Point(0, -4), Point(3, -4), Point(3, 0)]
    for p, ep in zip(pts, expected_points):
        assert math.isclose(p.x, ep.x, abs_tol=1e-15)
        assert math.isclose(p.y, ep.y, abs_tol=1e-15)


def test_rectangle_from_points():
    pts = [Point(0, 0), Point(2, 0), Point(2, 1), Point(0, 1)]
    rect = Rectangle.from_points(pts)
    assert rect.x == 0
    assert rect.y == 0
    assert rect.w == 2
    assert rect.h == 1
    assert rect.rot == 0


def test_rectangle_scale_around_center():
    rect = Rectangle.with_center(5, 5, 2, 4, rot=0)
    scaled = rect.scale_around_center(2)
    assert math.isclose(scaled.x + scaled.w / 2, 5, abs_tol=1e-15)
    assert math.isclose(scaled.y + scaled.h / 2, 5, abs_tol=1e-15)
    assert math.isclose(scaled.w, 4, abs_tol=1e-15)
    assert math.isclose(scaled.h, 8, abs_tol=1e-15)


def test_rotated_rectangle_scale_around_center():
    rect = Rectangle.with_center(5, 5, 2, 4, rot=math.pi / 4)
    scaled = rect.scale_around_center(2)

    center_orig = rect.points()[0] + rect.norm_vec_x(1) * (rect.w / 2) + rect.norm_vec_y(1) * (rect.h / 2)
    center_scaled = scaled.points()[0] + scaled.norm_vec_x(1) * (scaled.w / 2) + scaled.norm_vec_y(1) * (scaled.h / 2)

    assert math.isclose(center_orig.x, center_scaled.x, abs_tol=1e-10)
    assert math.isclose(center_orig.y, center_scaled.y, abs_tol=1e-10)


def test_rectangle_bounding_box():
    bbox = Rectangle(0, 0, 2, 2, rot=0).bounding_box()
    assert bbox.x == 0
    assert bbox.y == 0
    assert bbox.w == 2
    assert bbox.h == 2


def test_rotated_rectangle_bounding_box_values():
    rect = Rectangle(0, 0, 2, 2, rot=math.pi / 4)
    bbox = rect.bounding_box()

    expected_size = math.sqrt(2) * 2
    assert math.isclose(bbox.w, expected_size, abs_tol=1e-10)
    assert math.isclose(bbox.h, expected_size, abs_tol=1e-10)


def test_rectangle_intersection():
    r1 = Rectangle(0, 0, 2, 2)
    r2 = Rectangle(1, 1, 2, 2)
    inter = r1.intersection(r2)
    assert inter == Rectangle(1, 1, 1, 1)

    r3 = Rectangle(3, 3, 1, 1)
    inter_none = r1.intersection(r3)
    assert inter_none is None


def test_circle_bounding_box_and_scale():
    c = Circle(Point(5, 5), 3)
    bbox = c.bounding_box()
    assert bbox.x == 2
    assert bbox.y == 2
    assert bbox.w == 6
    assert bbox.h == 6

    scaled = c.scale(2)
    assert scaled.radius == 6
    assert scaled.center == Point(10, 10)


def test_ellipse_bounding_box_and_scale():
    e = Ellipse(Point(5, 5), (3, 5))
    bbox = e.bounding_box()
    assert bbox.x == 2
    assert bbox.y == 0
    assert bbox.w == 6
    assert bbox.h == 10

    scaled = e.scale(1.6)
    assert scaled.center == Point(8, 8)
    assert scaled.axes[0] == 3 * 1.6
    assert scaled.axes[1] == 8


def test_polygon_bounding_box_and_scale():
    poly = Polygon([Point(0, 0), Point(4, -2), Point(8, 0), Point(6, 4), Point(1, 4)])
    scaled = poly.scale(2)
    expected_points = [Point(0, 0), Point(8, -4), Point(16, 0), Point(12, 8), Point(2, 8)]
    for p, ep in zip(scaled.points, expected_points):
        assert p == ep

    expected_rect = Rectangle(0, -2, 8, 6)
    bbox = poly.bounding_box()
    assert expected_rect == bbox


def test_crop_and_paste_rect():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:5, 3:7] = 255
    rect = Rectangle(3, 2, 4, 3)
    patch = crop_rect_from_img(img, rect)
    assert patch.shape == (3, 4)
    assert np.all(patch == 255)

    img2 = np.zeros_like(img)
    paste_rect_into_img(img2, patch, rect)
    assert np.array_equal(img, img2)


def test_crop_and_paste_rectangle():
    img = np.zeros((20, 20), dtype=np.uint8)
    patch = np.full((3, 5), 255, dtype=np.uint8)

    rect = Rectangle(10, 10, 5, 3, rot=math.pi / 4)
    paste_rect_into_img(img, patch, rect)
    extracted = crop_rect_from_img(img, rect)
    assert extracted.shape == patch.shape
    assert np.count_nonzero(extracted) == (extracted.shape[0] * extracted.shape[1])

    rect = Rectangle(3, 2, 5, 3)
    paste_rect_into_img(img, patch, rect)
    extracted = crop_rect_from_img(img, rect)
    assert extracted.shape == patch.shape
    assert np.count_nonzero(extracted) == (extracted.shape[0] * extracted.shape[1])


def test_rectangle_translation():
    rect = Rectangle(1, 2, 3, 4)
    translated = rect.translate(5, 6)
    assert translated.x == rect.x + 5
    assert translated.y == rect.y + 6
    assert translated.w == rect.w
    assert translated.h == rect.h
    assert translated.rot == rect.rot


def test_circle_translation():
    circle = Circle(Point(1, 1), 5)
    translated = circle.translate(3, 4)
    assert translated.center == Point(circle.center.x + 3, circle.center.y + 4)
    assert translated.radius == circle.radius


def test_ellipse_translation():
    ellipse = Ellipse(Point(2, 3), (4, 5))
    translated = ellipse.translate(6, 7)
    assert translated.center == Point(ellipse.center.x + 6, ellipse.center.y + 7)
    assert translated.axes == ellipse.axes


def test_polygon_translation():
    points = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    poly = Polygon(points)
    translated = poly.translate(-10, -20)
    expected_points = [Point(p.x - 10, p.y - 20) for p in points]
    for p, ep in zip(translated.points, expected_points):
        assert p == ep


def test_rectangle_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    rect = Rectangle(2, 3, 4, 2)
    result = rectangle_mask(mask, rect)
    assert result.shape == mask.shape

    assert result[3, 2] == 255
    assert result[5, 2] == 255
    assert result[5, 6] == 255
    assert result[3, 6] == 255

    assert result[0, 0] == 0


def test_circle_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    circle = Circle(Point(5, 5), 2)
    result = circle_mask(mask, circle)

    assert result.shape == mask.shape

    assert result[5, 5] == 255

    assert result[3, 5] == 255
    assert result[7, 5] == 255
    assert result[5, 3] == 255
    assert result[5, 7] == 255

    assert result[0, 0] == 0


def test_ellipse_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    ellipse = Ellipse(Point(10, 10), (5, 3))
    result = ellipse_mask(mask, ellipse)

    assert result.shape == mask.shape

    assert result[10, 10] == 255

    assert result[7, 10] == 255
    assert result[13, 10] == 255
    assert result[10, 5] == 255
    assert result[10, 15] == 255

    assert result[0, 0] == 0


def test_polygon_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    poly = Polygon([Point(2, 2), Point(7, 2), Point(7, 7), Point(5, 9), Point(2, 7)])
    result = polygon_mask(mask, poly)

    assert result.shape == mask.shape

    assert result[2, 2] == 255
    assert result[7, 2] == 255
    assert result[7, 7] == 255
    assert result[9, 5] == 255
    assert result[2, 7] == 255

    assert result[3, 3] == 255

    assert result[0, 0] == 0


def test_crop_shape_from_img_rectangle_against_mask():
    img = np.full((10, 10), 100, dtype=np.uint8)
    rect = Rectangle(2, 2, 4, 4)

    cropped = crop_shape_from_img(img, rect, fill_value=0)
    mask = np.zeros_like(img, dtype=np.uint8)
    expected_mask = rectangle_mask(mask, rect)
    expected_cropped = np.where(expected_mask == 255, img, 0)

    np.testing.assert_array_equal(cropped, expected_cropped)


def test_crop_shape_from_img_circle():
    img = np.full((10, 10), 100, dtype=np.uint8)
    circle = Circle(Point(5, 5), 3)

    cropped = crop_shape_from_img(img, circle, fill_value=0)
    mask = np.zeros_like(img, dtype=np.uint8)
    expected_mask = circle_mask(mask, circle)
    expected_cropped = np.where(expected_mask == 255, img, 0)

    np.testing.assert_array_equal(cropped, expected_cropped)


def test_crop_shape_from_img_ellipse():
    img = np.full((20, 20), 100, dtype=np.uint8)
    ellipse = Ellipse(Point(10, 10), (5, 3))

    cropped = crop_shape_from_img(img, ellipse, fill_value=0)
    mask = np.zeros_like(img, dtype=np.uint8)
    expected_mask = ellipse_mask(mask, ellipse)
    expected_cropped = np.where(expected_mask == 255, img, 0)

    np.testing.assert_array_equal(cropped, expected_cropped)


def test_crop_shape_from_img_polygon():
    img = np.full((10, 10), 100, dtype=np.uint8)
    poly = Polygon([Point(2, 2), Point(7, 2), Point(7, 7), Point(2, 7)])

    cropped = crop_shape_from_img(img, poly, fill_value=0)
    mask = np.zeros_like(img, dtype=np.uint8)
    expected_mask = polygon_mask(mask, poly)
    expected_cropped = np.where(expected_mask == 255, img, 0)

    np.testing.assert_array_equal(cropped, expected_cropped)


def test_crop_shape_from_img_invalid_shape():
    img = np.zeros((10, 10), dtype=np.uint8)

    class DummyShape:
        pass

    dummy = DummyShape()

    with pytest.raises(ValueError, match="Unsupported shape type"):
        crop_shape_from_img(img, dummy)
