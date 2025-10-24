from dataclasses import dataclass
import numpy as np
from math import floor, ceil, pi, atan2
import cv2 as cv

Point = np.dtype([('x', np.float32), ('y', np.float32)])


def to_point(vec) -> Point:
    if hasattr(vec, 'x') and hasattr(vec, 'y'):
        return vec
    return np.rec.array((vec[0], vec[1]), dtype=Point)


def rotate_vector(vec: Point,
                  angle: float,
                  ) -> Point:
    """ rotate a vector by angle radians """
    vec = to_point(vec)
    return np.rec.array((vec.x * np.cos(angle) - vec.y * np.sin(angle),
                         vec.x * np.sin(angle) + vec.y * np.cos(angle)), dtype=Point)


def translate_point(point: Point,
                    vector: Point,
                    ) -> Point:
    """ translate a point by a vector """
    point = to_point(point)
    vector = to_point(vector)

    return np.rec.array((point.x + vector.x, point.y + vector.y), dtype=Point)


@dataclass
class Rectangle:
    x: float
    y: float
    w: float
    h: float
    rot: float = 0.0
    epsilon = 1e-6

    @staticmethod
    def with_center(cx: float,
                    cy: float,
                    w: float,
                    h: float,
                    rot: float = 0.0,
                    ):
        translation = rotate_vector((-w / 2, -h / 2), -rot)
        return Rectangle(cx + translation.x, cy + translation.y, w, h, rot)

    def scale(self,
              scale: float,
              ):
        return Rectangle(self.x * scale,
                         self.y * scale,
                         self.w * scale,
                         self.h * scale,
                         self.rot,
                         )

    def translate(self,
                  dx: float,
                  dy: float):
        return Rectangle(self.x + dx,
                         self.y + dy,
                         self.w,
                         self.h,
                         self.rot,
                         )

    def points(self) -> np.ndarray:
        """ return the four corners of the rectangle """

        ret = np.zeros(4, dtype=Point)
        # TODO this still returns the simple ndarray rather than array of Points

        ret[0] = (self.x, self.y)

        """ negative sign is taken since y values are increasing downwards"""
        vec1 = rotate_vector((self.w, 0), -self.rot)
        vec2 = rotate_vector((0, self.h), -self.rot)

        ret[1] = translate_point(ret[0], vec1)
        ret[2] = translate_point(ret[1], vec2)
        ret[3] = translate_point(ret[0], vec2)

        return ret

    def norm_vec_x(self) -> Point:
        return rotate_vector((1, 0), -self.rot)

    def norm_vec_y(self) -> Point:
        return rotate_vector((0, 1), -self.rot)

    def bounding_box(self):
        points = self.points()
        x = np.min(points[:]['x'])
        y = np.min(points[:]['y'])
        w = np.max(points[:]['x']) - x
        h = np.max(points[:]['y']) - y
        return Rectangle(x, y, w, h)

    def integer_boundary(self):
        """ return the integer boundary of the rectangle """
        bbox = self.bounding_box()
        int_x = floor(bbox.x)
        int_y = floor(bbox.y)

        return Rectangle(int_x, int_y, ceil((bbox.x + bbox.w) - int_x), ceil((bbox.y + bbox.h) - int_y))

    def area(self):
        return self.w * self.h

    def intersection(self,
                     other,
                     ):
        if self.rot != 0 or other.rot != 0:
            raise ValueError(
                "Intersection of rotated rectangles is not implemented")

        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        if x1 > x2 or y1 > y2:
            return None
        else:
            return Rectangle(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def from_points(cls,
                    points: np.ndarray,
                    ):
        # we assume that rectangle points are passed in the clockwise order
        # args:
        #  points: np.ndarray of shape (4, 2)

        assert points.shape == (4, 2)

        for i in range(4):
            vec = points[(i + 1) % 4] - points[i]
            angle = -atan2(vec[1], vec[0])

            if angle >= 0 and angle < pi / 2 + cls.epsilon:
                return Rectangle(points[i][0], points[i][1], np.linalg.norm(vec), np.linalg.norm(points[(i + 3) % 4] - points[i]), angle)

        assert False, f"Probably points {points} don't represent a rectangle"


def crop_rect_from_img(img: np.ndarray,
                       rect: Rectangle,
                       fill_value=255,
                       interpolation=cv.INTER_NEAREST,
                       ) -> np.ndarray:
    boundary = rect.integer_boundary().intersection(
        Rectangle(0, 0, img.shape[1], img.shape[0]))
    assert boundary is not None, "Rectangle is outside of the image"

    pts1 = np.float32([[point[0] - boundary.x, point[1] - boundary.y]
                       for point in rect.points()[:3]])

    pts2 = np.float32([[0, 0], [round(rect.w), 0], [round(rect.w), rect.h]])
    M = cv.getAffineTransform(pts1, pts2)

    border_value = fill_value if len(
        img.shape) == 2 else (fill_value,) * len(img.shape)

    return cv.warpAffine(img[boundary.y:boundary.y + boundary.h,
                             boundary.x:boundary.x + boundary.w], M, (round(rect.w), round(rect.h)),
                         borderMode=cv.BORDER_REPLICATE,
                         flags=interpolation,
                         borderValue=border_value,
                         ).astype(img.dtype)


def paste_rect_into_img(img: np.ndarray,
                        patch: np.ndarray,
                        rect: Rectangle,
                        interpolation=cv.INTER_NEAREST,
                        ) -> np.ndarray:
    boundary = rect.integer_boundary().intersection(
        Rectangle(0, 0, img.shape[1], img.shape[0]))
    assert boundary is not None, "Rectangle is outside of the image"

    pts1 = np.float32([[0, 0],
                       [patch.shape[1], 0],
                       [patch.shape[1], patch.shape[0]]],
                      )
    pts2 = np.float32([[point[0] - boundary.x,
                        point[1] - boundary.y]
                       for point in rect.points()[:3]])

    M = cv.getAffineTransform(pts1, pts2)

    transformed_patch = cv.warpAffine(patch,
                                      M,
                                      (boundary.w, boundary.h),
                                      flags=interpolation,
                                      )

    mask = cv.warpAffine(np.ones_like(patch),
                         M,
                         (boundary.w,
                          boundary.h,
                          ),
                         borderMode=cv.BORDER_CONSTANT,
                         borderValue=0,
                         flags=interpolation,
                         ).astype(bool)

    img[boundary.y:boundary.y + boundary.h,
        boundary.x:boundary.x + boundary.w][mask] = transformed_patch[mask]
    return img
