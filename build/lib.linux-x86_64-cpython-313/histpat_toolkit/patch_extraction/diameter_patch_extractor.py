import cv2
import numpy as np
import tcod

from ..geom import Rectangle
from ..tissue_detection.base_tissue_detector import BaseTissueDetector
from .base_patch_extractor import BasePatchExtractor


class DiameterPatchExtractor(BasePatchExtractor):
    def __init__(self, tissue_detector: BaseTissueDetector):
        super().__init__(tissue_detector=tissue_detector)

    def _extract_patches(
        self,
        mask: np.ndarray[np.uint8],
        scale: float | None,
        rect_width: int = 50,
        rect_height: int = 50,
        overlap: int = 10,
        **kwargs,
    ) -> list[Rectangle]:
        """
        This function draws a diameter of all the connected components in the mask
        and then tries to cover it using rectangles of given width and height

        Parameters:
            mask: np.ndarray
                Mask of detected tissue
            scale:
                The scale by which the original extract_patches(img) was scaled by
                this means, we may want to rescale some parameters in the anchorizer
                or other classes
            rect_width: int
                Width of the rectangles that will be used to cover the diameter
            rect_height: int
                Height of the rectangles that will be used to cover the diameter
            overlap: int
                How much overlap should the rectangles have (in pixels)
        """

        if scale is not None:
            rect_width = round(rect_width * scale)
            rect_height = round(rect_height * scale)
            overlap = round(overlap * scale)

        num_of_components, labels, _, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=4,
        )

        def furthest_point(
            graph,
            start,
        ):
            pf = tcod.path.Pathfinder(graph)
            pf.add_root(start)
            pf.resolve()
            dist = pf.distance.copy()
            dist[dist == np.iinfo(pf.distance.dtype).max] = 0
            end = np.unravel_index(dist.argmax(), dist.shape)
            return end, pf.path_to(end)

        ret = []

        for i in range(1, num_of_components):
            points = labels == i

            distance = cv2.distanceTransform(points.astype("uint8"), cv2.DIST_L2, 5)
            distance[distance != 0] = distance.max() + 1 - distance[distance != 0]

            graph = tcod.path.SimpleGraph(cost=distance.astype("int64"), cardinal=1, diagonal=1)
            point0 = np.unravel_index(distance.argmax(), distance.shape)
            point1, _ = furthest_point(graph, point0)
            point2, path = furthest_point(graph, point1)

            rectangles = []
            current = point1

            for i in range(1, len(path)):
                if i + 1 == len(path) or np.linalg.norm(path[i] - current) > rect_height:
                    point = path[i]
                    if i + 1 == len(path):
                        j = i - 1
                        while j > 0 and np.linalg.norm(path[j] - point) < rect_height:
                            j -= 1
                        current = path[j]
                    dx, dy = point - current
                    angle = -np.arctan2(dy, dx)

                    anchor = current - rect_width / 2 * np.array([np.sin(angle), np.cos(angle)])
                    rectangles.append(
                        Rectangle(anchor[1], anchor[0], rect_width, min(rect_height, np.linalg.norm((dx, dy))), -angle)
                    )

                    current = path[max(0, i - overlap)]

            ret.extend(rectangles)
            mask[points] = 0

        return ret
