from model_pipelines.base_model import BaseModel
from utils.thread_runner import ThreadRunner

THREADS_NUM = 10
MODEL_TYPE = "text"


class TextBaseModel(BaseModel):
    def __init__(self, model_name: str, texts: list, possible_tasks: list):
        super().__init__(model_type=MODEL_TYPE, model_name=model_name, possible_tasks=possible_tasks)
        self.texts = texts
        self.threads_num = len(texts) if len(texts) <= 10 else 10
        self.thread_runner = ThreadRunner(self.threads_num)
        self.model_pipeline = None
        self.task = None
        self.feature = None

    def run(self, task: str, feature: str):
        self.model_pipeline = self.init_pipeline(task)
        self.task = task
        self.feature = feature
        results = self.thread_runner.run_target_with_dask(self._predict_text, self.texts)
        return results

    def _predict_text(self, text):
        data = {"model": self.model_name, "task": self.task}
        if self.feature:
            data['feature'] = self.feature
            data["prediction"] = self.predict(self.model_pipeline, self.feature, text)
        else:
            data["prediction"] = self.predict(self.model_pipeline, text)
        return data

    def _predict_texts(self, texts):
        return [self._predict_text(text) for text in texts]
