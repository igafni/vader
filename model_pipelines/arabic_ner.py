from model_pipelines.text_base_model import TextBaseModel

MODEL_NAME = "arabic-ner"


class ArabicNer(TextBaseModel):
    def __init__(self, texts):
        super().__init__(MODEL_NAME, texts)

    def ner(self):
        return self.run('ner')
