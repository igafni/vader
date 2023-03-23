from fastapi import APIRouter
from routers.text_data import TextData
from model_pipelines.t5_base import T5Base

router = APIRouter(prefix='/models/text/t5_base', tags=["Text Models"])


@router.post('/summarization')
async def summarization(data: TextData):
    pipline_object = T5Base(data.texts)
    return pipline_object.summarization()
