from model_pipelines.text_base_model import TextBaseModel

MODEL_NAME = "opus-mt-ru-en"


class OpusMtRuEn(TextBaseModel):
    def __init__(self, texts):
        super().__init__(MODEL_NAME, texts)

    def translation(self):
        return self.run('translation')
