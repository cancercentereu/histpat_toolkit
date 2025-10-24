import numpy as np
import cv2 as cv

from .base_patch_extractor import BasePatchExtractor
from ..tissue_detection.base_tissue_detector import BaseTissueDetector
from .modifiers.base_tissue_modifier import BaseTissueModifier
from .anchorizers.base_tissue_anchorizer import BaseTissueAnchorizer
from .labelers.base_tissue_labeler import BaseTissueLabeler
from .rectangularizers.base_rectangularizer import BaseRectangularizer
from ..geom import Rectangle

from .anchor_patch_extractor_helper import get_labels_info


class BaseAnchorPatchExtractor(BasePatchExtractor):

    def __init__(self,
                 tissue_detector: BaseTissueDetector,
                 tissue_modifier: BaseTissueModifier,
                 tissue_anchorizer: BaseTissueAnchorizer,
                 tissue_labeler: BaseTissueLabeler,
                 tissue_rectangularizer: BaseRectangularizer,
                 max_area: int = 400*400,
                 ) -> None:
        """
            This class is a base class for patch extractors that will
            use the same framework for patch extraction. First we may
            modify the original tissue before we will try to find anchors
            that is centroids of the patches. Then using those anchors
            we label the rest of the original tissue and finally
            we rectangularize the areas with the same label.

            Parameters:
                tissue_detector (BaseTissueDetector):
                    used for detecting tissue in the image (0/1 mask)
                tissue_modifier (BaseTissueModifier):
                    modifies tissue before passing to anchorizer
                tissue_anchorizer (BaseTissueAnchorizer):
                    finds centroids (anchors) of the patches 
                tissue_labeler (BaseTissueLabeler):
                    used for labeling the original tissue using anchors
                tissue_rectangularizer (BaseRectangularizer):
                    used for covering the patch areas with the rectangles
                max_area (int): max area that the image will be scaled to
                    before passing to the model (default 400*400)
        """
        super().__init__(tissue_detector=tissue_detector,
                         max_area=max_area,
                         )
        self.tissue_modifier = tissue_modifier
        self.tissue_anchorizer = tissue_anchorizer
        self.tissue_labeler = tissue_labeler
        self.tissue_rectangularizer = tissue_rectangularizer

    def _extract_patches(self,
                         tissue_mask: np.ndarray[np.uint8],
                         scale: float = 1,
                         ) -> list[Rectangle]:
        modified_mask = self.tissue_modifier.modify_tissue(tissue_mask,
                                                           scale=scale,
                                                           )
        anchors_mask = self.tissue_anchorizer.anchorize_tissue(modified_mask,
                                                               scale=scale,
                                                               )
        labels = self.tissue_labeler.label_by_anchors_mask(anchors_mask,
                                                            tissue_mask,
                                                            scale=scale,
                                                            )

        lebels_info = get_labels_info(tissue_mask, labels.astype(np.int32))

        contours = [cv.findContours(((labels[min_y:max_y+1, min_x:max_x+1] == label) * 255).astype(np.uint8),
                                    cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE,
                                    offset=(min_x, min_y),
                                    )[0][0]
                    for label, (min_x, max_x, min_y, max_y) in enumerate(lebels_info)
                    if min_x != -1
                    ]

        return self.tissue_rectangularizer.rectangularize_tissue(contours,
                                                                 scale=scale,
                                                                 )
