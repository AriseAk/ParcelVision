class VolumeEstimator:

    def __init__(self):

        # depth heuristics per class
        self.depth_ratios = {
            "chair": 0.8,
            "sofa": 0.9,
            "table": 0.7,
            "cardboard box": 1.0,
            "carton box": 1.0,
            "shipping box": 1.0,
            "package": 1.0
        }


    def estimate_depth(self, label, width):

        ratio = self.depth_ratios.get(label, 1.0)
        return width * ratio


    def compute_volume(self, label, width, height):

        depth = self.estimate_depth(label, width)

        volume = width * height * depth

        return {
            "width": width,
            "height": height,
            "depth": depth,
            "volume_m3": volume
        }