from fastapi import APIRouter
from routers.text_data import TextData
from model_pipelines.text_model import TextModel
from model_files import ModelFiles
from model_pipelines.text_model import POSSIBLE_TASKS

router = APIRouter(prefix='/models/text', tags=["Text Models"])


@router.get("/all_models")
def get_all_models():
    model_files = ModelFiles()
    return {"models": model_files.get_model_list("text")}


@router.get("/all_tasks")
def get_all_tasks():
    return {"tasks": POSSIBLE_TASKS}


@router.post('/predict')
async def predict(data: TextData):
    pipline_object = TextModel(model_name=data.model_name, texts=data.texts)
    return pipline_object.predict_task(task=data.task, feature=data.feature)
