from abc import ABC, abstractmethod
from transformers import pipeline
import os
from model_files import ModelFiles


class BaseModel(ABC):
    def __init__(self, model_type: str, model_name: str, possible_tasks: list):
        self.model_name = model_name
        self.model_type = model_type
        self.model_location = self._model_location(model_name)
        self.possible_tasks = possible_tasks
        self.possible_models = self.get_model_files()
        self.check_model()

    def _model_location(self, model_name):
        return os.path.join("models", self.model_type, model_name)

    def init_pipeline(self, task):
        model_pipeline = pipeline(task, model=self.model_location)
        return model_pipeline

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def predict(model_pipeline, *features):
        model_score = model_pipeline(*features)
        return model_score

    def predict_with_init_pipeline(self, task, *features):
        model_pipeline = self.init_pipeline(task)
        model_score = model_pipeline(*features)
        return model_score

    def check_task(self, task):
        if task not in self.possible_tasks:
            raise Exception("Wrong Model Task")

    def check_model(self):
        if self.model_name not in self.possible_models:
            raise Exception("Wrong Model Name")

    def get_model_files(self):
        model_files = ModelFiles()
        return model_files.get_model_list(self.model_type)

    def get_possible_tasks(self):
        return self.possible_tasks
