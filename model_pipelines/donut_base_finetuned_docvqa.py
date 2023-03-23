from model_pipelines.base_model import BaseModel
from model_pipelines.image_base_model import ImageBaseModel

MODEL_NAME = "donut-base-finetuned-docvqa"


class DonutBaseFinetunedDocvQa(ImageBaseModel):
    def __init__(self, images):
        super().__init__(MODEL_NAME, images)

    def document_question_answering(self, question: str):
        return self.run('document-question-answering', question)



