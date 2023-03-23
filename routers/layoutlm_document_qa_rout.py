from fastapi import APIRouter
from routers.image_data import ImageData
from model_pipelines.layoutlm_document_qa import LayoutlmDocumentQa

router = APIRouter(prefix='/models/image/layoutlm', tags=["Image Models"])


@router.post('/document-question-answering')
async def document_question_answering(data: ImageData):
    pipline_object = LayoutlmDocumentQa(data.images)
    return pipline_object.document_question_answering(question=data.feature)
