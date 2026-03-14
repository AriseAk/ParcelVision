class DepthFusion:

    def __init__(self):

        # approximate laptop / phone camera intrinsics
        self.fx = 600
        self.fy = 600
        self.cx = 320
        self.cy = 240


    def pixel_to_world(self, center, depth):

        u, v = center
        Z = depth

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        return {
            "x": float(X),
            "y": float(Y),
            "z": float(Z)
        }


    def bbox_to_dimensions(self, bbox, depth):

        x1, y1, x2, y2 = bbox

        pixel_width = x2 - x1
        pixel_height = y2 - y1

        W = (pixel_width * depth) / self.fx
        H = (pixel_height * depth) / self.fy

        return {
            "width": float(W),
            "height": float(H)
        }