from histpat_toolkit.image_pyramid import ImagePyramid
from histpat_toolkit.geom import Rectangle

pyramid = ImagePyramid(
  levels=14,
  width=8000,
  height=6000,
  tile_size=512,
  tiles_url="https://example.com/path/{level}/{x}_{y}.jpeg"
)
scale = 1/2
region_of_interest = Rectangle(4100, 1900, 1000, 1200).scale(scale)
arr = pyramid.crop_rect(region_of_interest, scale=scale)

# Example: save to file with OpenCV
import cv2
img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
cv2.imwrite("example.jpg", img)