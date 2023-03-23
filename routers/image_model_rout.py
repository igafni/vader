from fastapi import APIRouter
from routers.image_data import ImageData
from model_pipelines.image_model import ImageModel
from model_files import ModelFiles
from model_pipelines.image_model import POSSIBLE_TASKS

router = APIRouter(prefix='/models/image', tags=["Image Models"])


@router.get("/all_models")
def get_all_models():
    model_files = ModelFiles()
    return {"models": model_files.get_model_list("image")}


@router.get("/all_tasks")
def get_all_tasks():
    return {"tasks": POSSIBLE_TASKS}


@router.post('/predict')
async def predict(data: ImageData):
    pipline_object = ImageModel(model_name=data.model_name, images=data.images)
    return pipline_object.predict_task(task=data.task, feature=data.feature)
