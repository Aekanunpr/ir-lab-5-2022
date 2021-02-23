from fastapi import FastAPI, File, UploadFile
from pythainlp.tokenize import word_tokenize
import io
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/text/{text}")
def read_word(text: str):
    return {"word": text)}