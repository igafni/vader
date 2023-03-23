from fastapi import FastAPI
import uvicorn
from routers.layoutlm_document_qa_rout import router as layoutlm_document_qa_router
from routers.donut_base_finetuned_docvqa_rout import router as donut_base_finetuned_docvqa_router
from routers.opus_mt_ru_en_rout import router as opus_mt_ru_en_router
from routers.t5_base_rout import router as t5_base_router
from routers.text_model_rout import router as text_model_router
from routers.image_model_rout import router as image_model_router

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse

local_host = "http://192.168.92.1:8080"


description = """
 ![Vader](http://192.168.92.1:8080/static/vader_logo.png)
 
* ## AI Infrastructure Model Engine Platform
* ## The engine can run and finetune huggingface models
## Image model post request example:

    import base64
    import json
    import requests
    
    ## Required Image Format -  im_b64:  
    def format_convert(image_bytes)
        im_b64 = base64.b64encode(image_bytes).decode("utf8")
        return im_b64
    
    def convert_image(image_path:str):
        with open(image_file, "rb") as f:
            image_bytes = f.read()
        return format_convert(image_bytes)
        
    def post_images_to_api(api:str, images_path:list, model_name:str, task:str, feature:str):
        images = [convert_image(image_path) for image_path in images_path]
        payload = json.dumps({"images": images, "model_name": model_name, "task": task, "feature": feature})
        response = requests.post(api, data=payload)
        try:
            data = response.json()
            print(data)
        except requests.exceptions.RequestException:
            print(response.text)
            
    # Example Post:  
          
    post_images_to_api(api="http://vader:8080/models/image/predict",
            images_path=[r"images/invoice-template-us-neat-750px.png"], 
            model_name="layoutlmv2-base-uncased-finetuned-docvqa",
            task="document-question-answering", feature="What is the invoice number?")
    

## Text model post request example:

    def post_texts_to_api(api:str, texts:list, model_name:str, task:str, feature:str = None):
        data = {"texts": texts, "model_name": model_name, "task": task}
        if feature: # Only for text models that required feature
            data.update({"feature": feature})
        payload = json.dumps(data)
        response = requests.post(api, data=payload)
        try:
            data = response.json()
            print(data)
        except requests.exceptions.RequestException:
            print(response.text)
            
    # Example Post:  
         
    post_texts_to_api(api="http://vader:8080/models/text/predict",
            texts=["Меня зовут Вольфганг и я живу в Берлине","порядком, основанным на правилах"], 
            model_name="opus-mt-ru-en",
            task="translation")
                    \n

        """



app = FastAPI(title="Vader API", description=description, docs_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/static/vader_logo.png")
async def read_image():
    return FileResponse("/static/vader_logo.png")


app.include_router(text_model_router)
app.include_router(image_model_router)


# An endpoint to serve images for documentation


@app.get("/docs", include_in_schema=False)
async def swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Vader API", swagger_favicon_url="/static/vader.png")


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8080, log_level="info")
