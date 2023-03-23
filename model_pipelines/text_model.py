from model_pipelines.text_base_model import TextBaseModel

POSSIBLE_TASKS = ['translation', 'summarization', 'question-answering',"text-classification"]


class TextModel(TextBaseModel):
    def __init__(self, model_name, texts):
        super().__init__(model_name, texts, possible_tasks=POSSIBLE_TASKS)

    def predict_task(self, task: str, feature: str = None):
        self.check_task(task)
        return self.run(task, feature)
