from model_pipelines.text_base_model import TextBaseModel

MODEL_NAME = "t5-base"


class T5Base(TextBaseModel):
    def __init__(self, texts):
        super().__init__(MODEL_NAME, texts)

    def summarization(self):
        return self.run('summarization')
