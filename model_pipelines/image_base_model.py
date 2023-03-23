from model_pipelines.base_model import BaseModel
import base64
import io
from PIL import Image
from utils.thread_runner import ThreadRunner

MODEL_TYPE = "image"
THREADS_NUM = 10


class ImageBaseModel(BaseModel):
    def __init__(self, model_name: str, images: list, possible_tasks: list):
        super().__init__(model_type=MODEL_TYPE, model_name=model_name, possible_tasks=possible_tasks)
        self.images = [self.convert_to_pil_object(image) for image in images]
        self.threads_num = len(images) if len(images) <= 10 else 10
        self.thread_runner = ThreadRunner(self.threads_num)
        self.model_pipeline = None
        self.task = None
        self.feature = None

    @staticmethod
    def convert_to_pil_object(image):
        img_bytes = base64.b64decode(image.encode('utf-8'))
        return Image.open(io.BytesIO(img_bytes))

    def _predict_image(self, image):
        data = {"model": self.model_name, "task": self.task}
        if self.feature:
            data['feature'] = self.feature
            data["prediction"] = self.predict(self.model_pipeline, image, self.feature)
        else:
            data["prediction"] = self.predict(self.model_pipeline, image)
        return data

    def _predict_images(self, images):
        return [self._predict_image(image) for image in images]

    def run(self, task: str, feature: str):
        self.model_pipeline = self.init_pipeline(task)
        self.task = task
        self.feature = feature
        results = self.thread_runner.run_target_with_dask(self._predict_image, self.images)
        return results
