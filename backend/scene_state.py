from spatial_mapper import SpatialMapper


class SceneStateManager:

    def __init__(self):

        self.objects = {}
        self.max_missing_frames = 30

        # spatial mapper
        self.mapper = SpatialMapper()

    def update(self, inventory, frame_index):

        for obj in inventory:

            obj_id = obj["object_id"]

            camera_position = self.mapper.pixel_to_camera(obj["center"])

            if obj_id not in self.objects:

                self.objects[obj_id] = {

                    "id": obj_id,
                    "label": obj["label"],
                    "confidence": obj["confidence"],

                    "first_seen": frame_index,
                    "last_seen": frame_index,

                    "bbox": obj["bbox"],
                    "center": obj["center"],
                    "area": obj["area"],

                    "stability": obj["stability_score"],

                    "camera_position": camera_position,

                    "world_position": {
                        "x": None,
                        "y": None,
                        "z": None
                    }
                }

            else:

                obj_state = self.objects[obj_id]

                obj_state["last_seen"] = frame_index
                obj_state["bbox"] = obj["bbox"]
                obj_state["center"] = obj["center"]
                obj_state["area"] = obj["area"]
                obj_state["confidence"] = obj["confidence"]
                obj_state["stability"] = obj["stability_score"]

                obj_state["camera_position"] = camera_position


        # remove disappeared objects
        to_remove = []

        for obj_id, obj_state in self.objects.items():

            if frame_index - obj_state["last_seen"] > self.max_missing_frames:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.objects[obj_id]

        return self.objects