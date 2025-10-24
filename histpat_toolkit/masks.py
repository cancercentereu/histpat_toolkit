import cv2
import numpy as np
import svgwrite
from PIL import ImageColor


def _build_colors_array(colors) -> np.ndarray:
    """
    Convert a color map into a NumPy array of RGB values.

    Args:
        color_map (ColorMap): Object containing color values.

    Returns:
        np.ndarray: Array of shape (N, 4) containing RGBA colors as uint8.
    """
    colors_values = [color.value for color in colors]
    colors_array = np.array([ImageColor.getrgb(c) for c in colors_values], dtype=np.uint8)
    return colors_array


def mask_to_contours(
    mask_image: np.ndarray,
    colors,
    approx_epsilon_ratio: float = 0.1,
    approximation: bool = False,
) -> list[tuple[list[np.ndarray], tuple[int, int, int, int]]]:
    """
    Extract contours from a mask image for each color in the color map.

    Args:
        mask_image (np.ndarray): RGBA mask image (H, W, C).
        colors: Iterable of color objects, each with a `.value` attribute
                compatible with PIL.ImageColor.getrgb.

    Returns:
        list of tuples:
            Each tuple has the structure:
            (list_of_contours, rgba_color_tuple)

            - list_of_contours (list[np.ndarray]): List of contours for that color,
              where each contour is an array of shape (N, 1, 2).
            - rgba_color_tuple (tuple[int, int, int, int]): RGBA color tuple.
    """
    hsv_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2HSV)
    color_array = _build_colors_array(colors)

    mask_contours = []
    for color in color_array:
        hsv_color = cv2.cvtColor(np.uint8([[color[:3]]]), cv2.COLOR_RGB2HSV)[0][0]
        hsv_mask = cv2.inRange(hsv_image, hsv_color, hsv_color)

        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if approximation:
            approx_contours: list[np.ndarray] = []
            for cnt in contours:
                epsilon = approx_epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx = approx.reshape((-1, 1, 2)).astype(np.int32)
                approx_contours.append(approx)
            
            if len(approx_contours) > 0:
                mask_contours.append((approx_contours, tuple(int(c) for c in color)))
        else:
            if len(contours) > 0:
                mask_contours.append((contours, tuple(int(c) for c in color)))

    return mask_contours


def single_color_to_svg(
    contours_list: list[np.ndarray], color: tuple[int, int, int, int], shape: tuple, display_width: int = 400
) -> svgwrite.Drawing:
    """
    Convert a single color's contours into an SVG drawing.

    Args:
        contours_list: list of contours (from cv2.findContours)
        color: RGBA tuple
        shape: mask image shape (height, width, channels)
        display_width: SVG width in pixels
    """
    height, width = shape[:2]
    display_height = int(display_width * height / width)

    dwg = svgwrite.Drawing(size=(f"{display_width}px", f"{display_height}px"), viewBox=f"0 0 {width} {height}")

    rgb_hex = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
    alpha = int(color[3]) / 255 if len(color) == 4 else 1.0

    for contour in contours_list:
        points = contour.squeeze(axis=1)  # (N, 2)
        path_data = "M " + " ".join(f"{int(x)},{int(y)}" for x, y in points) + " Z"
        dwg.add(dwg.path(d=path_data, fill=rgb_hex, fill_opacity=alpha, fill_rule="evenodd"))

    return dwg


def masks_to_svg(
    all_masks_contours: list[tuple[list[np.ndarray], tuple[int, int, int, int]]],
    shape: tuple,
    display_width: int = 400,
) -> svgwrite.Drawing:
    """
    Combine multiple masks' contours into a single SVG drawing.

    Args:
        all_masks_contours: list of mask_to_contours outputs
        shape: mask image shape (height, width, channels)
        display_width: width of SVG in pixels
    """
    height, width = shape[:2]
    display_height = int(display_width * height / width)

    combined_dwg = svgwrite.Drawing(size=(f"{display_width}px", f"{display_height}px"), viewBox=f"0 0 {width} {height}")

    for mask_contours in all_masks_contours:
        single_dwg = single_color_to_svg(mask_contours[0], mask_contours[1], shape, display_width)
        for element in single_dwg.elements:
            combined_dwg.add(element)

    return combined_dwg
