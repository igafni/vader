from model_pipelines.image_base_model import ImageBaseModel

POSSIBLE_TASKS = ['document-question-answering']


class ImageModel(ImageBaseModel):
    def __init__(self, model_name:str, images:list):
        super().__init__(model_name, images, possible_tasks=POSSIBLE_TASKS)

    def predict_task(self, task: str, feature: str = None):
        self.check_task(task)
        return self.run(task, feature)
