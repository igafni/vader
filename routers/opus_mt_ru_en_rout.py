from fastapi import APIRouter
from routers.text_data import TextData
from model_pipelines.opus_mt_ru_en import OpusMtRuEn

router = APIRouter(prefix='/models/text/opus', tags=["Text Models"])


@router.post('/translation')
async def translation(data: TextData):
    pipline_object = OpusMtRuEn(data.texts)
    return pipline_object.translation()
