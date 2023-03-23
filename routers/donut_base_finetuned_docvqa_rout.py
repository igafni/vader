from fastapi import APIRouter
from routers.image_data import ImageData
from model_pipelines.donut_base_finetuned_docvqa import DonutBaseFinetunedDocvQa

router = APIRouter(prefix='/models/image/donut', tags=["Image Models"])


@router.post('/document-question-answering')
async def document_question_answering(data: ImageData):
    pipline_object = DonutBaseFinetunedDocvQa(data.images)
    return pipline_object.document_question_answering(question=data.feature)
