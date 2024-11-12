from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse,JSONResponse, FileResponse
from PIL import Image
import numpy as np
from io import BytesIO
import glob
import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm

from utils import preprocess_image, postprocess_mask

app = FastAPI()

# Load your U-Net model with custom loss function
MODEL_PATH = './Unet-efficientnetb2.weights.h5'
BACKBONE = 'efficientnetb2'

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=8, activation='softmax')
model.load_weights(MODEL_PATH) 

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Segmentation API"}


@app.get("/list_files")
async def list_files():
    # Set a default folder if none is provided

    search_folder = "."

    # Define patterns to search for images and masks
    image_pattern = os.path.join(search_folder, '*left*')
    mask_pattern = os.path.join(search_folder, '*_cat*')

    # Get lists of image and mask filenames
    image_filenames = [os.path.basename(path) for path in glob.glob(image_pattern)]
    mask_filenames = [os.path.basename(path) for path in glob.glob(mask_pattern)]

    # Return just the lists of filenames
    return JSONResponse(content={
        "images": image_filenames,
        "masks": mask_filenames
    })

@app.get("/get_file")
async def get_file(filename: str, file_type: str = Query("image")):
    # Set the folder based on file type
    folder = "."
    file_path = os.path.join(folder, filename)
    
    # Check if file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return the file as a response
    return FileResponse(file_path)

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Preprocess the image
    input_image = preprocess_image(image)

    # Predict the mask
    prediction = model.predict(input_image)
    mask = postprocess_mask(prediction)
    
    return JSONResponse(content={"mask": mask.tolist()})