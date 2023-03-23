import os

MODELS_DIRECTORY = "models"


class ModelFiles(object):
    def __init__(self):
        pass

    @staticmethod
    def _model_location(model_type):
        return os.path.join(".", MODELS_DIRECTORY, model_type)

    def get_model_list(self, model_type):
        model_location = self._model_location(model_type)
        models = os.listdir(model_location)
        return models
