from shapely.geometry import Polygon, Point


class Geofence:

    def __init__(self, normalized_points):
        self.normalized_points = normalized_points

    def to_pixel_points(self, frame_width, frame_height):

        return [
            (
                int(x * frame_width),
                int(y * frame_height)
            )
            for x, y in self.normalized_points
        ]

    def polygon_pixels(self, frame_width, frame_height):

        pixel_points = self.to_pixel_points(
            frame_width,
            frame_height
        )

        return Polygon(pixel_points)

    def is_inside(self, bbox, frame_width, frame_height):

        x1, y1, x2, y2 = bbox

        # Find center of bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        polygon = self.polygon_pixels(
            frame_width,
            frame_height
        )

        point = Point(cx, cy)

        return polygon.contains(point)