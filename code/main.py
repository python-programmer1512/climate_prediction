# 192.168.0.2
# uvicorn main:app --reload
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

from starlette.middleware.cors import CORSMiddleware

import numpy as np
import uvicorn
import cv2

import image_prediction as imgp

app = FastAPI()

origins = [
    "http://localhost:8000"    # 또는 "http://localhost:5173"
]

@app.post("/image/")
async def create_file(request : Request):
    async with request.form(max_part_size=20*1024*1024) as form:
        file: UploadFile = form["file"]
        content = await file.read()
        nparr = np.frombuffer(content,np.uint8)
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite("./imgdata2.jpg",img)
        print("save")
        ans,percent = imgp.Synthesis_img_mask_bypath(path="./imgdata2.jpg")
        print(f"ans : {ans}, percent : {percent}")
        
        return ans,percent



app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



if __name__ == "__main__":
    uvicorn.run(app)
    #uvicorn.run(app, host="0.0.0.0")# -> 배포시