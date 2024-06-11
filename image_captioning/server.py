import PIL
import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from code.CaptionGenerator import CaptionGenerator

app = FastAPI()

caption_generator = CaptionGenerator(
    rnn_model_place='./data/caption_en_model40.model',
    cnn_model_place='./data/ResNet50.model',
    dictonary_place='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
    beamsize=3,
    depth_limit=50,
    gpu_id=-1,
    first_word='<sos>',
)

@app.post("/process_image")
async def process_image(image: UploadFile = File(...)):
    # Read the image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_image_path = 'temp_image.jpg'
    cv2.imwrite(temp_image_path, img)

    # Generate captions
    captions = caption_generator.generate(temp_image_path)
    text = " ".join(captions[0]["sentence"][1:-1])

    return JSONResponse(content={"text": text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
