from fastapi import FastAPI, File, UploadFile
import PIL
from PIL import Image
import botnoi.resnet50 as rn
import shutil
from botnoi import cv
import os
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from io import BytesIO
app = FastAPI()

# output function
modFile = 'mymod.mod'
mod = pickle.load(open(modFile,'rb'))


@app.post("/image")
async def image(image: UploadFile = File(...)):
    #file#Image.open(image.file)
    with open("file.png",'wb') as f:
        shutil.copyfileobj(image.file, f)
    feat = rn.extract_feature("file.png")
    res = mod.predict([feat])[0]
    return {"class":res}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/text/{text}")
def read_word(text: str):
    return {"word": text}


if __name__ == '__main__':
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=80, debug=True) 