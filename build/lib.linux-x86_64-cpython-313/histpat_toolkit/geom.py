from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import atan2, ceil, floor, pi
from typing import Any, Iterable, Self, Sequence

import cv2 as cv
import numpy as np


class Point(np.ndarray[Any, np.dtype[np.float64]]):
    """
    n-dimensional point used for locations.
    inherits +, -, * (as dot-product)
    > p1 = Point(1, 2)
    > p2 = Point(4, 5)
    > p1 + p2
    Point([5, 7])
    """

    def __new__(cls, x=0.0, y=0.0):
        """
        :param cls:
        :param input_array: Defaults to 2d origin
        """
        obj = np.asarray((x, y)).view(cls)
        return obj

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other) -> Self:
        return super().__add__(other)

    def __sub__(self, other) -> Self:
        return super().__sub__(other)  # type: ignore

    def __mul__(self, other) -> Self:
        return super().__mul__(other)  # type: ignore

    def __truediv__(self, other) -> Self:
        return super().__truediv__(other)  # type: ignore

    def __floordiv__(self, other) -> Self:
        return super().__floordiv__(other)  # type: ignore

    def __pow__(self, other) -> Self:
        return super().__pow__(other)  # type: ignore

    def norm(self, ord=None):
        return np.linalg.norm(self, ord=ord)

    def dist(self, other):
        """
        Both points must have the same dimensions
        :return: Euclidean distance
        """
        return (self - other).norm()

    def rotate(self, angle):
        return Point(
            self.x * np.cos(angle) - self.y * np.sin(angle),
            self.x * np.sin(angle) + self.y * np.cos(angle),
        )


PointLike = Point | Iterable[float]


def to_point(pt: PointLike):
    if isinstance(pt, Point):
        return pt
    return Point(*pt)


def rotate_vector(
    vec: Point | tuple[float, float],
    angle: float,
) -> Point:
    return to_point(vec).rotate(angle)


def translate_point(
    point: Point | tuple[float, float],
    vector: Point | tuple[float, float],
) -> Point:
    """translate a point by a vector"""
    return to_point(point) + to_point(vector)


class Shape(ABC):
    @abstractmethod
    def bounding_box(self) -> "Rectangle":
        """
        Returns the smallest non-rotated rectangle that contains the shape.
        """
        pass

    @abstractmethod
    def rotated_bounding_box(self) -> "Rectangle":
        """
        Returns the smallest (possibly rotated) rectangle that contains the shape.
        """
        pass

    @abstractmethod
    def scale(self, scale: float) -> Self:
        pass

    @abstractmethod
    def translate(dx: float, dy: float) -> Self:
        """
        Translate the shape by dx and dy.
        """
        pass


@dataclass
class Rectangle(Shape):
    x: float
    y: float
    w: float
    h: float
    rot: float = 0.0
    epsilon = 1e-6

    @staticmethod
    def with_center(
        cx: float,
        cy: float,
        w: float,
        h: float,
        rot: float = 0.0,
    ):
        delta = rotate_vector((w, h), -rot) / 2
        return Rectangle(
            cx - delta.x,
            cy - delta.y,
            w,
            h,
            rot,
        )

    def clone(self):
        return Rectangle(self.x, self.y, self.w, self.h, self.rot)

    def scale(
        self,
        scale: float,
    ):
        return Rectangle(
            self.x * scale,
            self.y * scale,
            self.w * scale,
            self.h * scale,
            self.rot,
        )

    def scale_around_center(self, scale: float):
        rect = self.clone()
        diff = rotate_vector((rect.w, rect.h), -rect.rot)

        rect.x -= diff.x * (scale - 1) / 2
        rect.y -= diff.y * (scale - 1) / 2
        rect.w *= scale
        rect.h *= scale
        return rect

    def translate(self, dx: float, dy: float):
        return Rectangle(
            self.x + dx,
            self.y + dy,
            self.w,
            self.h,
            self.rot,
        )

    def points(self) -> list[Point]:
        """return the four corners of the rectangle"""

        ret = [Point(0, 0) for _ in range(4)]

        ret[0] = Point(self.x, self.y)

        """ negative sign is taken since y values are increasing downwards"""
        vec1 = rotate_vector(Point(self.w, 0), -self.rot)
        vec2 = rotate_vector(Point(0, self.h), -self.rot)

        ret[1] = ret[0] + vec1
        ret[2] = ret[0] + vec1 + vec2
        ret[3] = ret[0] + vec2

        return ret

    def norm_vec_x(
        self,
        len: float = 1,
    ) -> Point:
        return rotate_vector((len, 0), -self.rot)

    def norm_vec_y(
        self,
        len: float = 1,
    ) -> Point:
        return rotate_vector((0, len), -self.rot)

    def bounding_box(self):
        points = self.points()
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        x = np.min(xs)
        y = np.min(ys)
        w = np.max(xs) - x
        h = np.max(ys) - y
        return Rectangle(x, y, w, h)

    def rotated_bounding_box(self):
        return self.clone()

    def integer_boundary(self):
        """return the integer boundary of the rectangle"""
        bbox = self.bounding_box()
        int_x = floor(bbox.x)
        int_y = floor(bbox.y)

        return Rectangle(int_x, int_y, ceil((bbox.x + bbox.w) - int_x), ceil((bbox.y + bbox.h) - int_y))

    def area(self):
        return self.w * self.h

    def intersection(
        self,
        other,
    ):
        if self.rot != 0 or other.rot != 0:
            raise ValueError("Intersection of rotated rectangles is not implemented")

        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        if x1 > x2 or y1 > y2:
            return None
        else:
            return Rectangle(x1, y1, x2 - x1, y2 - y1)

    def global_point_to_local(self, point: PointLike) -> Point:
        return (to_point(point) - Point(self.x, self.y)).rotate(self.rot)

    def local_point_to_global(self, point: PointLike) -> Point:
        return to_point(point).rotate(-self.rot) + Point(self.x, self.y)

    @classmethod
    def from_points(
        cls,
        points: Sequence[PointLike],
    ):
        # we assume that rectangle points are passed in the clockwise order
        # args:
        #  points: collection of 4 points

        assert len(points) == 4
        points = [to_point(p) for p in points]

        for i in range(4):
            vec = points[(i + 1) % 4] - points[i]
            angle = -atan2(vec[1], vec[0])

            if angle >= 0 and angle < pi / 2 + cls.epsilon:
                return Rectangle(
                    points[i][0],
                    points[i][1],
                    vec.norm(),
                    (points[(i + 3) % 4] - points[i]).norm(),
                    angle,
                )

        assert False, f"Probably points {points} don't represent a rectangle"


@dataclass
class Polygon(Shape):
    points: list[Point]

    def __post_init__(self):
        self.points = [to_point(p) for p in self.points]

    def scale(self, scale: float):
        return Polygon([p * scale for p in self.points])

    def bounding_box(self):
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)

        width = max_x - min_x
        height = max_y - min_y

        return Rectangle(min_x, min_y, width, height)

    def rotated_bounding_box(self):
        rect = cv.minAreaRect(np.array(self.points, dtype=np.float32))
        cx, cy = rect[0]
        w, h = rect[1]
        rot = rect[2]
        if rot == 90.0:
            w, h = h, w
            rot = 0.0
        return Rectangle.with_center(
            cx,
            cy,
            w,
            h,
            rot * pi / 180,
        )

    def translate(self, dx: float, dy: float):
        return Polygon([(p.x + dx, p.y + dy) for p in self.points])


@dataclass
class Circle(Shape):
    center: Point
    radius: float

    def __post_init__(self):
        self.center = to_point(self.center)

    def bounding_box(self):
        return Rectangle(self.center.x - self.radius, self.center.y - self.radius, 2 * self.radius, 2 * self.radius)

    def rotated_bounding_box(self):
        return self.bounding_box()

    def scale(self, scale: float):
        return Circle(self.center * scale, self.radius * scale)

    def translate(self, dx: float, dy: float):
        return Circle((self.center.x + dx, self.center.y + dy), self.radius)


@dataclass
class Ellipse(Shape):
    center: Point
    axes: tuple[float, float]

    def __post_init__(self):
        self.center = to_point(self.center)

    def bounding_box(self):
        return Rectangle(self.center.x - self.axes[0], self.center.y - self.axes[1], 2 * self.axes[0], 2 * self.axes[1])

    def rotated_bounding_box(self):
        return self.bounding_box()

    def scale(self, scale: float):
        return Ellipse(self.center * scale, (self.axes[0] * scale, self.axes[1] * scale))

    def translate(self, dx: float, dy: float):
        return Ellipse((self.center.x + dx, self.center.y + dy), self.axes)


def crop_shape_from_img(img: np.ndarray, shape: Shape, fill_value=255) -> np.ndarray:
    mask_picker = {
        Rectangle: rectangle_mask,
        Circle: circle_mask,
        Ellipse: ellipse_mask,
        Polygon: polygon_mask,
    }

    mask_func = mask_picker.get(type(shape))
    if mask_func is None:
        raise ValueError(f"Unsupported shape type: {type(shape)}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = mask_func(mask, shape)
    img = img.copy()
    img[~(mask.astype(bool))] = fill_value
    return img


def rectangle_mask(mask: np.ndarray, rect: Rectangle):
    cv.fillPoly(mask, [np.array(rect.points(), dtype=np.int32)], (255, 0, 0))
    return mask


def circle_mask(mask: np.ndarray, circle: Circle) -> np.ndarray:
    cv.circle(mask, (int(circle.center.x), int(circle.center.y)), int(circle.radius), (255, 0, 0), -1)
    return mask


def ellipse_mask(mask: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    cv.ellipse(
        mask,
        (int(ellipse.center.x), int(ellipse.center.y)),
        (int(ellipse.axes[0]), int(ellipse.axes[1])),
        0,
        0,
        360,
        (255, 0, 0),
        -1,
    )
    return mask


def polygon_mask(mask: np.ndarray, polygon: Polygon) -> np.ndarray:
    cv.fillPoly(mask, [np.array([p.tolist() for p in polygon.points], dtype=np.int32)], color=(255, 0, 0))
    return mask


def crop_rect_from_img(
    img: np.ndarray,
    rect: Rectangle,
    fill_value=255,
    interpolation=cv.INTER_NEAREST,
) -> np.ndarray:
    boundary = rect.integer_boundary().intersection(Rectangle(0, 0, img.shape[1], img.shape[0]))
    assert boundary is not None, "Rectangle is outside of the image"

    pts1 = np.float32([[point.x - boundary.x, point.y - boundary.y] for point in rect.points()[:3]])

    pts2 = np.float32([[0, 0], [round(rect.w), 0], [round(rect.w), rect.h]])
    M = cv.getAffineTransform(pts1, pts2)

    border_value = fill_value if len(img.shape) == 2 else (fill_value,) * len(img.shape)

    return cv.warpAffine(
        img[boundary.y : boundary.y + boundary.h, boundary.x : boundary.x + boundary.w],
        M,
        (round(rect.w), round(rect.h)),
        borderMode=cv.BORDER_REPLICATE,
        flags=interpolation,
        borderValue=border_value,
    ).astype(img.dtype)


def paste_rect_into_img(
    img: np.ndarray,
    patch: np.ndarray,
    rect: Rectangle,
    interpolation=cv.INTER_NEAREST,
    maximum=False,
) -> None:
    boundary = rect.integer_boundary().intersection(Rectangle(0, 0, img.shape[1], img.shape[0]))
    assert boundary is not None, "Rectangle is outside of the image"

    pts1 = np.float32(
        [[0, 0], [patch.shape[1], 0], [patch.shape[1], patch.shape[0]]],
    )
    pts2 = np.float32([[point[0] - boundary.x, point[1] - boundary.y] for point in rect.points()[:3]])

    M = cv.getAffineTransform(pts1, pts2)

    transformed_patch = cv.warpAffine(
        patch,
        M,
        (boundary.w, boundary.h),
        flags=interpolation,
    )

    mask = cv.warpAffine(
        np.ones_like(patch),
        M,
        (
            boundary.w,
            boundary.h,
        ),
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
        flags=interpolation,
    ).astype(bool)

    if maximum:
        img[boundary.y : boundary.y + boundary.h, boundary.x : boundary.x + boundary.w][mask] = np.maximum(
            img[boundary.y : boundary.y + boundary.h, boundary.x : boundary.x + boundary.w][mask],
            transformed_patch[mask],
        )
    else:
        img[boundary.y : boundary.y + boundary.h, boundary.x : boundary.x + boundary.w][mask] = transformed_patch[mask]
