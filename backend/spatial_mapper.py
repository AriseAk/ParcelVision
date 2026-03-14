class SpatialMapper:

    def __init__(self, width=640, height=480):

        # camera center
        self.cx = width / 2
        self.cy = height / 2

    def pixel_to_camera(self, center):

        u, v = center

        # normalize coordinates relative to camera center
        x = (u - self.cx) / self.cx
        y = (v - self.cy) / self.cy

        return {
            "x": float(x),
            "y": float(y),
            "z": None
        }