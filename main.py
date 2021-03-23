from fastapi import FastAPI, File, UploadFile, Form
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
import requests
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

@app.get("/imageurl")
async def imgurl(p_image_url: str):
    #file#Image.open(image.file)
    r = requests.get(p_image_url, allow_redirects=True)
    with open("file2.png",'wb') as f:
        f.write(r.content)
    feat = rn.extract_feature("file2.png")
    res = mod.predict([feat])[0]
    return {"class":str(res)}


@app.get("/text")
def read_word(text: str):
    return {"word": text}


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == '__main__':
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8080, debug=True) 
