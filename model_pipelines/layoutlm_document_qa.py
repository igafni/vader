from model_pipelines.image_base_model import ImageBaseModel

MODEL_NAME = "layoutlm-document-qa"


class LayoutlmDocumentQa(ImageBaseModel):
    def __init__(self, images):
        super().__init__(MODEL_NAME, images)

    def document_question_answering(self, question: str):
        return self.run('document-question-answering', question)
